import os
import gc
import psutil
import tracemalloc
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """
    Comprehensive memory profiler for tracking memory usage during tiling operations.
    
    Tracks:
    - Python heap memory (via tracemalloc)
    - System memory (RSS, VMS via psutil)
    - Open file descriptors
    - Memory snapshots at key points
    - Memory growth between operations
    """
    
    def __init__(self, output_dir: str = None, enable_tracemalloc: bool = True):
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.profile_dir = self.output_dir / "memory_profiles"
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_tracemalloc = enable_tracemalloc
        self.snapshots = []
        self.start_time = datetime.now()
        self.process = psutil.Process()
        
        # Initialize tracemalloc if enabled
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
        
        # Track baseline metrics
        self.baseline_memory = self.get_current_memory()
        self.baseline_fds = self.count_open_fds()
        
        logger.info(f"Memory profiler initialized. Baseline RSS: {self.baseline_memory['rss_mb']:.1f}MB")
    
    def get_current_memory(self) -> Dict[str, Any]:
        """Get current memory usage from multiple sources."""
        memory_info = self.process.memory_info()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'open_fds': self.count_open_fds()
        }
        
        # Add tracemalloc metrics if available
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            metrics['tracemalloc_current_mb'] = current / 1024 / 1024
            metrics['tracemalloc_peak_mb'] = peak / 1024 / 1024
        
        return metrics
    
    def count_open_fds(self) -> int:
        """Count open file descriptors."""
        try:
            return len(self.process.open_files())
        except:
            # Fallback for systems where open_files() might not work
            try:
                fd_path = f"/proc/{self.process.pid}/fd"
                if os.path.exists(fd_path):
                    return len(os.listdir(fd_path))
            except:
                pass
        return -1
    
    def take_snapshot(self, label: str, image_name: str = "", force_gc: bool = True) -> Dict[str, Any]:
        """
        Take a memory snapshot with detailed metrics.
        
        Args:
            label: Description of the snapshot point
            image_name: Optional image name being processed
            force_gc: Whether to force garbage collection before snapshot
        """
        if force_gc:
            gc.collect()
        
        current_memory = self.get_current_memory()
        
        # Calculate deltas from baseline
        delta_rss = current_memory['rss_mb'] - self.baseline_memory['rss_mb']
        delta_fds = current_memory['open_fds'] - self.baseline_fds
        
        snapshot = {
            'label': label,
            'image_name': image_name,
            'memory': current_memory,
            'delta_rss_mb': delta_rss,
            'delta_fds': delta_fds,
            'snapshot_index': len(self.snapshots)
        }
        
        # Get tracemalloc top stats if available
        if tracemalloc.is_tracing():
            snapshot['top_allocations'] = self.get_top_allocations(limit=10)
        
        self.snapshots.append(snapshot)
        
        # Log significant changes
        if abs(delta_rss) > 100:  # More than 100MB change
            logger.warning(f"Significant memory change at '{label}': {delta_rss:+.1f}MB")
        
        if delta_fds > 10:  # More than 10 new file descriptors
            logger.warning(f"File descriptor increase at '{label}': {delta_fds:+d} FDs")
        
        return snapshot
    
    def get_top_allocations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top memory allocations from tracemalloc."""
        if not tracemalloc.is_tracing():
            return []
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:limit]
        
        allocations = []
        for stat in top_stats:
            allocations.append({
                'file': stat.traceback.format()[0],
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })
        
        return allocations
    
    def compare_snapshots(self, start_label: str, end_label: str) -> Dict[str, Any]:
        """Compare two snapshots to analyze memory growth."""
        start_snap = next((s for s in self.snapshots if s['label'] == start_label), None)
        end_snap = next((s for s in self.snapshots if s['label'] == end_label), None)
        
        if not start_snap or not end_snap:
            return {}
        
        comparison = {
            'start': start_label,
            'end': end_label,
            'rss_growth_mb': end_snap['memory']['rss_mb'] - start_snap['memory']['rss_mb'],
            'fd_growth': end_snap['memory']['open_fds'] - start_snap['memory']['open_fds'],
            'time_delta': end_snap['memory']['timestamp'] - start_snap['memory']['timestamp']
        }
        
        return comparison
    
    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """Generate a comprehensive memory profile report."""
        report = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'baseline_memory': self.baseline_memory,
            'final_memory': self.get_current_memory(),
            'total_snapshots': len(self.snapshots),
            'snapshots': self.snapshots,
            'memory_timeline': self._create_timeline(),
            'peak_memory': self._find_peak_memory(),
            'suspicious_growth_points': self._find_suspicious_growth()
        }
        
        if save_to_file:
            report_path = self.profile_dir / f"memory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Memory report saved to {report_path}")
        
        return report
    
    def _create_timeline(self) -> List[Dict[str, Any]]:
        """Create a timeline of memory usage."""
        timeline = []
        for i, snapshot in enumerate(self.snapshots):
            timeline.append({
                'index': i,
                'label': snapshot['label'],
                'image': snapshot['image_name'],
                'rss_mb': snapshot['memory']['rss_mb'],
                'delta_rss_mb': snapshot['delta_rss_mb']
            })
        return timeline
    
    def _find_peak_memory(self) -> Dict[str, Any]:
        """Find peak memory usage point."""
        if not self.snapshots:
            return {}
        
        peak_snapshot = max(self.snapshots, key=lambda s: s['memory']['rss_mb'])
        return {
            'label': peak_snapshot['label'],
            'image': peak_snapshot['image_name'],
            'rss_mb': peak_snapshot['memory']['rss_mb'],
            'snapshot_index': peak_snapshot['snapshot_index']
        }
    
    def _find_suspicious_growth(self, threshold_mb: float = 500) -> List[Dict[str, Any]]:
        """Find points where memory grew suspiciously."""
        suspicious = []
        
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]
            
            growth = curr['memory']['rss_mb'] - prev['memory']['rss_mb']
            
            if growth > threshold_mb:
                suspicious.append({
                    'from': prev['label'],
                    'to': curr['label'],
                    'growth_mb': growth,
                    'image': curr['image_name']
                })
        
        return suspicious
    
    def log_summary(self):
        """Log a summary of memory usage."""
        current = self.get_current_memory()
        logger.info(f"Memory Summary - RSS: {current['rss_mb']:.1f}MB "
                   f"(+{current['rss_mb'] - self.baseline_memory['rss_mb']:.1f}MB), "
                   f"FDs: {current['open_fds']} (+{current['open_fds'] - self.baseline_fds})")