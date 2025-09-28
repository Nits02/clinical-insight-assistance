"""
Memory Management Module for Clinical Insights Assistant

This module provides memory management capabilities for storing and retrieving:
- Clinical trial data
- Analysis results
- Generated insights
- Agent state and history
- Learned patterns and knowledge
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import pickle
import sqlite3
import os
from pathlib import Path
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Data class representing a memory entry."""
    entry_id: str
    entry_type: str
    content: Any
    metadata: Dict[str, Any]
    created_at: datetime
    accessed_at: datetime
    access_count: int
    importance_score: float
    tags: List[str]


class MemoryManager:
    """
    Memory management system for the Clinical Insights Assistant.
    Provides persistent storage and retrieval of data, insights, and learned knowledge.
    """
    
    def __init__(self, memory_dir: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the Memory Manager.
        
        Args:
            memory_dir (str, optional): Directory for storing memory files. Uses MEMORY_DIR from .env if not provided.
            config (Dict, optional): Configuration dictionary. Uses environment variables if not provided.
        """
        # Use environment variable if memory_dir is not provided
        self.memory_dir = Path(memory_dir or os.getenv('MEMORY_DIR', 'memory'))
        self.memory_dir.mkdir(exist_ok=True)
        
        self.config = config or self._get_default_config()
        
        # Initialize storage systems
        sqlite_path = self.config['sqlite_config']['database_path']
        self.db_path = Path(sqlite_path) if os.path.isabs(sqlite_path) else self.memory_dir / "memory.db"
        self.data_dir = self.memory_dir / "data"
        self.insights_dir = self.memory_dir / "insights"
        self.models_dir = self.memory_dir / "models"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.insights_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # In-memory caches
        self.data_cache = {}
        self.insight_cache = {}
        self.pattern_cache = {}
        
        # Memory statistics
        self.stats = {
            'total_entries': 0,
            'data_entries': 0,
            'insight_entries': 0,
            'pattern_entries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'last_cleanup': datetime.now()
        }
        
        logger.info(f"Memory Manager initialized with directory: {memory_dir}")
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration for memory management from environment variables.
        
        Returns:
            Dict: Configuration parameters loaded from .env file.
        """
        # Helper function to convert string boolean to bool
        def str_to_bool(value: str) -> bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        
        return {
            # Cache settings
            'cache_config': {
                'max_cache_size': int(os.getenv('MEMORY_CACHE_MAX_SIZE', '100')),
                'cache_ttl_hours': int(os.getenv('MEMORY_CACHE_TTL_HOURS', '24')),
                'auto_cleanup_interval': int(os.getenv('MEMORY_AUTO_CLEANUP_INTERVAL', '3600')),
                'memory_limit_mb': int(os.getenv('MEMORY_LIMIT_MB', '500'))
            },
            
            # Storage settings
            'storage_config': {
                'compress_data': str_to_bool(os.getenv('MEMORY_COMPRESS_DATA', 'true')),
                'backup_enabled': str_to_bool(os.getenv('MEMORY_BACKUP_ENABLED', 'true')),
                'backup_interval_hours': int(os.getenv('MEMORY_BACKUP_INTERVAL_HOURS', '6')),
                'max_backup_files': int(os.getenv('MEMORY_MAX_BACKUP_FILES', '10')),
                'data_retention_days': int(os.getenv('MEMORY_DATA_RETENTION_DAYS', '90'))
            },
            
            # Importance scoring
            'importance_config': {
                'base_importance': float(os.getenv('MEMORY_BASE_IMPORTANCE', '0.5')),
                'access_weight': float(os.getenv('MEMORY_ACCESS_WEIGHT', '0.3')),
                'recency_weight': float(os.getenv('MEMORY_RECENCY_WEIGHT', '0.2')),
                'content_weight': float(os.getenv('MEMORY_CONTENT_WEIGHT', '0.5')),
                'decay_factor': float(os.getenv('MEMORY_DECAY_FACTOR', '0.95'))
            },
            
            # Pattern learning
            'pattern_config': {
                'min_pattern_frequency': int(os.getenv('MEMORY_MIN_PATTERN_FREQUENCY', '3')),
                'pattern_confidence_threshold': float(os.getenv('MEMORY_PATTERN_CONFIDENCE_THRESHOLD', '0.7')),
                'max_patterns_per_type': int(os.getenv('MEMORY_MAX_PATTERNS_PER_TYPE', '50')),
                'pattern_update_interval': int(os.getenv('MEMORY_PATTERN_UPDATE_INTERVAL', '3600'))
            },
            
            # SQLite settings
            'sqlite_config': {
                'database_path': os.getenv('SQLITE_DATABASE_PATH', 'memory/memory.db'),
                'timeout': int(os.getenv('SQLITE_TIMEOUT', '30')),
                'isolation_level': os.getenv('SQLITE_ISOLATION_LEVEL', 'DEFERRED'),
                'check_same_thread': str_to_bool(os.getenv('SQLITE_CHECK_SAME_THREAD', 'false'))
            }
        }
    
    def _init_database(self):
        """Initialize the SQLite database for metadata storage."""
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        sqlite_config = self.config['sqlite_config']
        with sqlite3.connect(
            self.db_path,
            timeout=sqlite_config['timeout'],
            isolation_level=sqlite_config['isolation_level'],
            check_same_thread=sqlite_config['check_same_thread']
        ) as conn:
            cursor = conn.cursor()
            
            # Create memory entries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_entries (
                    entry_id TEXT PRIMARY KEY,
                    entry_type TEXT NOT NULL,
                    file_path TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP,
                    accessed_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    importance_score REAL DEFAULT 0.5,
                    tags TEXT
                )
            ''')
            
            # Create insights table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS insights (
                    insight_id TEXT PRIMARY KEY,
                    insight_type TEXT,
                    title TEXT,
                    description TEXT,
                    confidence_score REAL,
                    clinical_significance TEXT,
                    created_at TIMESTAMP,
                    patient_ids TEXT,
                    cohorts TEXT,
                    tags TEXT
                )
            ''')
            
            # Create patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    pattern_data TEXT,
                    frequency INTEGER,
                    confidence REAL,
                    last_seen TIMESTAMP,
                    created_at TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entry_type ON memory_entries(entry_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memory_entries(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memory_entries(importance_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_insight_type ON insights(insight_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_type ON patterns(pattern_type)')
            
            conn.commit()
    
    def store_data(self, data_id: str, data: Any, metadata: Optional[Dict] = None, 
                   tags: Optional[List[str]] = None) -> str:
        """
        Store data in memory with metadata.
        
        Args:
            data_id (str): Unique identifier for the data.
            data (Any): Data to store.
            metadata (Dict, optional): Additional metadata.
            tags (List[str], optional): Tags for categorization.
            
        Returns:
            str: Entry ID for the stored data.
        """
        entry_id = self._generate_entry_id(data_id, 'data')
        current_time = datetime.now()
        
        # Store data to file
        file_path = self.data_dir / f"{entry_id}.pkl"
        
        try:
            with open(file_path, 'wb') as f:
                if self.config['storage_config']['compress_data']:
                    import gzip
                    with gzip.open(f, 'wb') as gz_f:
                        pickle.dump(data, gz_f)
                else:
                    pickle.dump(data, f)
            
            # Store metadata in database
            metadata = metadata or {}
            metadata.update({
                'original_id': data_id,  # Store the original ID for lookup
                'data_type': type(data).__name__,
                'data_shape': getattr(data, 'shape', None),
                'data_size': len(data) if hasattr(data, '__len__') else None
            })
            
            self._store_entry_metadata(
                entry_id=entry_id,
                entry_type='data',
                file_path=str(file_path),
                metadata=metadata,
                created_at=current_time,
                tags=tags or []
            )
            
            # Update cache
            self.data_cache[entry_id] = {
                'data': data,
                'cached_at': current_time,
                'access_count': 0
            }
            
            # Update statistics
            self.stats['total_entries'] += 1
            self.stats['data_entries'] += 1
            
            logger.info(f"Data stored with entry ID: {entry_id}")
            return entry_id
            
        except Exception as e:
            logger.error(f"Failed to store data {data_id}: {str(e)}")
            raise
    
    def get_data(self, data_id: str) -> Any:
        """
        Retrieve data from memory.
        
        Args:
            data_id (str): Data identifier or entry ID.
            
        Returns:
            Any: Retrieved data.
        """
        # Try to find entry ID
        entry_id = data_id if data_id.startswith('data_') else self._find_entry_id(data_id, 'data')
        
        if not entry_id:
            raise KeyError(f"Data not found: {data_id}")
        
        # Check cache first
        if entry_id in self.data_cache:
            cache_entry = self.data_cache[entry_id]
            cache_entry['access_count'] += 1
            self.stats['cache_hits'] += 1
            self._update_access_metadata(entry_id)
            return cache_entry['data']
        
        # Load from file
        try:
            file_path = self._get_file_path(entry_id)
            
            with open(file_path, 'rb') as f:
                if self.config['storage_config']['compress_data']:
                    import gzip
                    with gzip.open(f, 'rb') as gz_f:
                        data = pickle.load(gz_f)
                else:
                    data = pickle.load(f)
            
            # Update cache
            current_time = datetime.now()
            self.data_cache[entry_id] = {
                'data': data,
                'cached_at': current_time,
                'access_count': 1
            }
            
            # Update access metadata
            self._update_access_metadata(entry_id)
            
            self.stats['cache_misses'] += 1
            return data
            
        except Exception as e:
            logger.error(f"Failed to retrieve data {data_id}: {str(e)}")
            raise
    
    def store_insight(self, insight) -> str:
        """
        Store an insight in memory.
        
        Args:
            insight: Insight object to store.
            
        Returns:
            str: Entry ID for the stored insight.
        """
        entry_id = self._generate_entry_id(insight.insight_id, 'insight')
        current_time = datetime.now()
        
        # Store insight to file
        file_path = self.insights_dir / f"{entry_id}.json"
        
        try:
            insight_dict = asdict(insight)
            # Convert datetime objects to ISO strings
            insight_dict['created_at'] = insight.created_at.isoformat()
            
            with open(file_path, 'w') as f:
                json.dump(insight_dict, f, indent=2, default=str)
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO insights 
                    (insight_id, insight_type, title, description, confidence_score, 
                     clinical_significance, created_at, patient_ids, cohorts, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    insight.insight_id,
                    insight.insight_type,
                    insight.title,
                    insight.description,
                    insight.confidence_score,
                    insight.clinical_significance,
                    insight.created_at,
                    json.dumps(insight.patient_ids),
                    json.dumps(insight.cohorts),
                    json.dumps(insight.tags)
                ))
                conn.commit()
            
            # Store metadata
            metadata = {
                'insight_type': insight.insight_type,
                'confidence_score': insight.confidence_score,
                'clinical_significance': insight.clinical_significance,
                'patient_count': len(insight.patient_ids),
                'cohort_count': len(insight.cohorts)
            }
            
            self._store_entry_metadata(
                entry_id=entry_id,
                entry_type='insight',
                file_path=str(file_path),
                metadata=metadata,
                created_at=current_time,
                tags=insight.tags
            )
            
            # Update cache
            self.insight_cache[entry_id] = {
                'insight': insight,
                'cached_at': current_time,
                'access_count': 0
            }
            
            # Update statistics
            self.stats['total_entries'] += 1
            self.stats['insight_entries'] += 1
            
            logger.info(f"Insight stored with entry ID: {entry_id}")
            return entry_id
            
        except Exception as e:
            logger.error(f"Failed to store insight {insight.insight_id}: {str(e)}")
            raise
    
    def get_insights(self, insight_type: Optional[str] = None, 
                    patient_ids: Optional[List[str]] = None,
                    tags: Optional[List[str]] = None,
                    min_confidence: Optional[float] = None,
                    limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve insights based on filters.
        
        Args:
            insight_type (str, optional): Filter by insight type.
            patient_ids (List[str], optional): Filter by patient IDs.
            tags (List[str], optional): Filter by tags.
            min_confidence (float, optional): Minimum confidence score.
            limit (int, optional): Maximum number of results.
            
        Returns:
            List[Dict]: List of matching insights.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM insights WHERE 1=1"
            params = []
            
            if insight_type:
                query += " AND insight_type = ?"
                params.append(insight_type)
            
            if min_confidence:
                query += " AND confidence_score >= ?"
                params.append(min_confidence)
            
            if tags:
                for tag in tags:
                    query += " AND tags LIKE ?"
                    params.append(f'%"{tag}"%')
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            columns = [desc[0] for desc in cursor.description]
            insights = []
            
            for row in rows:
                insight_dict = dict(zip(columns, row))
                
                # Parse JSON fields
                insight_dict['patient_ids'] = json.loads(insight_dict['patient_ids'])
                insight_dict['cohorts'] = json.loads(insight_dict['cohorts'])
                insight_dict['tags'] = json.loads(insight_dict['tags'])
                
                # Filter by patient_ids if specified
                if patient_ids:
                    if not any(pid in insight_dict['patient_ids'] for pid in patient_ids):
                        continue
                
                insights.append(insight_dict)
            
            return insights
    
    def store_pattern(self, pattern_id: str, pattern_type: str, pattern_data: Dict,
                     frequency: int = 1, confidence: float = 0.5) -> str:
        """
        Store a learned pattern.
        
        Args:
            pattern_id (str): Unique pattern identifier.
            pattern_type (str): Type of pattern.
            pattern_data (Dict): Pattern data.
            frequency (int): Pattern frequency.
            confidence (float): Pattern confidence.
            
        Returns:
            str: Entry ID for the stored pattern.
        """
        entry_id = self._generate_entry_id(pattern_id, 'pattern')
        current_time = datetime.now()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if pattern exists
                cursor.execute('SELECT frequency FROM patterns WHERE pattern_id = ?', (pattern_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing pattern
                    new_frequency = existing[0] + frequency
                    cursor.execute('''
                        UPDATE patterns 
                        SET frequency = ?, confidence = ?, last_seen = ?
                        WHERE pattern_id = ?
                    ''', (new_frequency, confidence, current_time, pattern_id))
                else:
                    # Insert new pattern
                    cursor.execute('''
                        INSERT INTO patterns 
                        (pattern_id, pattern_type, pattern_data, frequency, confidence, last_seen, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        pattern_id, pattern_type, json.dumps(pattern_data),
                        frequency, confidence, current_time, current_time
                    ))
                
                conn.commit()
            
            # Update cache
            self.pattern_cache[pattern_id] = {
                'pattern_type': pattern_type,
                'pattern_data': pattern_data,
                'frequency': frequency,
                'confidence': confidence,
                'cached_at': current_time
            }
            
            # Update statistics
            if not existing:
                self.stats['total_entries'] += 1
                self.stats['pattern_entries'] += 1
            
            logger.info(f"Pattern stored: {pattern_id}")
            return entry_id
            
        except Exception as e:
            logger.error(f"Failed to store pattern {pattern_id}: {str(e)}")
            raise
    
    def get_patterns(self, pattern_type: Optional[str] = None,
                    min_frequency: Optional[int] = None,
                    min_confidence: Optional[float] = None,
                    limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve learned patterns.
        
        Args:
            pattern_type (str, optional): Filter by pattern type.
            min_frequency (int, optional): Minimum frequency.
            min_confidence (float, optional): Minimum confidence.
            limit (int, optional): Maximum number of results.
            
        Returns:
            List[Dict]: List of matching patterns.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM patterns WHERE 1=1"
            params = []
            
            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type)
            
            if min_frequency:
                query += " AND frequency >= ?"
                params.append(min_frequency)
            
            if min_confidence:
                query += " AND confidence >= ?"
                params.append(min_confidence)
            
            query += " ORDER BY frequency DESC, confidence DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            columns = [desc[0] for desc in cursor.description]
            patterns = []
            
            for row in rows:
                pattern_dict = dict(zip(columns, row))
                pattern_dict['pattern_data'] = json.loads(pattern_dict['pattern_data'])
                patterns.append(pattern_dict)
            
            return patterns
    
    def search_memory(self, query: str, entry_types: Optional[List[str]] = None,
                     limit: int = 10) -> List[Dict]:
        """
        Search memory entries by text query.
        
        Args:
            query (str): Search query.
            entry_types (List[str], optional): Filter by entry types.
            limit (int): Maximum number of results.
            
        Returns:
            List[Dict]: Search results.
        """
        results = []
        
        # Search insights
        if not entry_types or 'insight' in entry_types:
            insights = self.get_insights(limit=limit * 2)  # Get more to filter
            
            for insight in insights:
                # Simple text matching
                searchable_text = f"{insight['title']} {insight['description']} {' '.join(insight['tags'])}"
                if query.lower() in searchable_text.lower():
                    results.append({
                        'entry_type': 'insight',
                        'entry_id': insight['insight_id'],
                        'title': insight['title'],
                        'description': insight['description'],
                        'relevance_score': self._calculate_relevance(query, searchable_text),
                        'created_at': insight['created_at']
                    })
        
        # Search patterns
        if not entry_types or 'pattern' in entry_types:
            patterns = self.get_patterns(limit=limit * 2)
            
            for pattern in patterns:
                searchable_text = f"{pattern['pattern_type']} {json.dumps(pattern['pattern_data'])}"
                if query.lower() in searchable_text.lower():
                    results.append({
                        'entry_type': 'pattern',
                        'entry_id': pattern['pattern_id'],
                        'title': f"Pattern: {pattern['pattern_type']}",
                        'description': f"Frequency: {pattern['frequency']}, Confidence: {pattern['confidence']:.2f}",
                        'relevance_score': self._calculate_relevance(query, searchable_text),
                        'created_at': pattern['created_at']
                    })
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:limit]
    
    def cleanup_memory(self, force: bool = False):
        """
        Clean up old and low-importance memory entries.
        
        Args:
            force (bool): Force cleanup regardless of interval.
        """
        current_time = datetime.now()
        
        # Check if cleanup is needed
        if not force:
            last_cleanup = self.stats['last_cleanup']
            cleanup_interval = timedelta(seconds=self.config['cache_config']['auto_cleanup_interval'])
            
            if current_time - last_cleanup < cleanup_interval:
                return
        
        logger.info("Starting memory cleanup")
        
        # Clean up cache
        self._cleanup_cache()
        
        # Clean up old entries
        retention_days = self.config['storage_config']['data_retention_days']
        cutoff_date = current_time - timedelta(days=retention_days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Find old entries
            cursor.execute('''
                SELECT entry_id, file_path FROM memory_entries 
                WHERE created_at < ? AND importance_score < 0.3
            ''', (cutoff_date,))
            
            old_entries = cursor.fetchall()
            
            for entry_id, file_path in old_entries:
                try:
                    # Remove file
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                    
                    # Remove from database
                    cursor.execute('DELETE FROM memory_entries WHERE entry_id = ?', (entry_id,))
                    
                    # Remove from cache
                    self.data_cache.pop(entry_id, None)
                    self.insight_cache.pop(entry_id, None)
                    
                except Exception as e:
                    logger.warning(f"Failed to cleanup entry {entry_id}: {str(e)}")
            
            conn.commit()
        
        # Update statistics
        self.stats['last_cleanup'] = current_time
        
        logger.info(f"Memory cleanup completed. Removed {len(old_entries)} old entries")
    
    def _cleanup_cache(self):
        """Clean up in-memory caches."""
        current_time = datetime.now()
        cache_ttl = timedelta(hours=self.config['cache_config']['cache_ttl_hours'])
        max_cache_size = self.config['cache_config']['max_cache_size']
        
        # Clean up data cache
        expired_keys = []
        for key, cache_entry in self.data_cache.items():
            if current_time - cache_entry['cached_at'] > cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.data_cache[key]
        
        # Limit cache size
        if len(self.data_cache) > max_cache_size:
            # Remove least recently used entries
            sorted_entries = sorted(
                self.data_cache.items(),
                key=lambda x: x[1]['access_count']
            )
            
            entries_to_remove = len(self.data_cache) - max_cache_size
            for key, _ in sorted_entries[:entries_to_remove]:
                del self.data_cache[key]
        
        # Similar cleanup for insight cache
        expired_keys = []
        for key, cache_entry in self.insight_cache.items():
            if current_time - cache_entry['cached_at'] > cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.insight_cache[key]
    
    def _generate_entry_id(self, base_id: str, entry_type: str) -> str:
        """
        Generate a unique entry ID.
        
        Args:
            base_id (str): Base identifier.
            entry_type (str): Type of entry.
            
        Returns:
            str: Unique entry ID.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hash_suffix = hashlib.md5(f"{base_id}_{timestamp}".encode()).hexdigest()[:8]
        return f"{entry_type}_{timestamp}_{hash_suffix}"
    
    def _find_entry_id(self, base_id: str, entry_type: str) -> Optional[str]:
        """
        Find entry ID by base ID and type.
        
        Args:
            base_id (str): Base identifier.
            entry_type (str): Entry type.
            
        Returns:
            Optional[str]: Entry ID if found.
        """
        # First check if we stored it with this exact base_id
        sqlite_config = self.config['sqlite_config']
        with sqlite3.connect(
            self.db_path,
            timeout=sqlite_config['timeout'],
            check_same_thread=sqlite_config['check_same_thread']
        ) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT entry_id FROM memory_entries 
                WHERE entry_type = ? AND (metadata LIKE ? OR entry_id LIKE ?)
                ORDER BY created_at DESC LIMIT 1
            ''', (entry_type, f'%"original_id": "{base_id}"%', f'%{base_id}%'))
            
            result = cursor.fetchone()
            return result[0] if result else None
    
    def _get_file_path(self, entry_id: str) -> str:
        """
        Get file path for an entry ID.
        
        Args:
            entry_id (str): Entry ID.
            
        Returns:
            str: File path.
        """
        sqlite_config = self.config['sqlite_config']
        with sqlite3.connect(
            self.db_path,
            timeout=sqlite_config['timeout'],
            check_same_thread=sqlite_config['check_same_thread']
        ) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT file_path FROM memory_entries WHERE entry_id = ?', (entry_id,))
            
            result = cursor.fetchone()
            if not result:
                raise KeyError(f"Entry not found: {entry_id}")
            
            return result[0]
    
    def _store_entry_metadata(self, entry_id: str, entry_type: str, file_path: str,
                             metadata: Dict, created_at: datetime, tags: List[str]):
        """
        Store entry metadata in database.
        
        Args:
            entry_id (str): Entry ID.
            entry_type (str): Entry type.
            file_path (str): File path.
            metadata (Dict): Metadata dictionary.
            created_at (datetime): Creation timestamp.
            tags (List[str]): Tags list.
        """
        sqlite_config = self.config['sqlite_config']
        with sqlite3.connect(
            self.db_path,
            timeout=sqlite_config['timeout'],
            check_same_thread=sqlite_config['check_same_thread']
        ) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO memory_entries 
                (entry_id, entry_type, file_path, metadata, created_at, accessed_at, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry_id, entry_type, file_path, json.dumps(metadata),
                created_at, created_at, json.dumps(tags)
            ))
            conn.commit()
    
    def _update_access_metadata(self, entry_id: str):
        """
        Update access metadata for an entry.
        
        Args:
            entry_id (str): Entry ID.
        """
        current_time = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE memory_entries 
                SET accessed_at = ?, access_count = access_count + 1,
                    importance_score = importance_score * 1.1
                WHERE entry_id = ?
            ''', (current_time, entry_id))
            conn.commit()
    
    def _calculate_relevance(self, query: str, text: str) -> float:
        """
        Calculate relevance score for search results.
        
        Args:
            query (str): Search query.
            text (str): Text to match against.
            
        Returns:
            float: Relevance score (0-1).
        """
        query_words = query.lower().split()
        text_lower = text.lower()
        
        # Simple relevance calculation
        matches = sum(1 for word in query_words if word in text_lower)
        return matches / len(query_words) if query_words else 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict[str, Any]: Memory statistics.
        """
        # Update database statistics
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM memory_entries')
            total_entries = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM insights')
            insight_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM patterns')
            pattern_count = cursor.fetchone()[0]
        
        # Calculate cache statistics
        cache_size = len(self.data_cache) + len(self.insight_cache) + len(self.pattern_cache)
        
        # Calculate disk usage
        total_size = 0
        for dir_path in [self.data_dir, self.insights_dir, self.models_dir]:
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        return {
            'total_entries': total_entries,
            'insight_count': insight_count,
            'pattern_count': pattern_count,
            'cache_size': cache_size,
            'cache_hit_rate': self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0,
            'disk_usage_mb': total_size / (1024 * 1024),
            'last_cleanup': self.stats['last_cleanup'].isoformat()
        }
    
    def backup_memory(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the memory system.
        
        Args:
            backup_path (str, optional): Custom backup path.
            
        Returns:
            str: Path to the backup file.
        """
        if not backup_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.memory_dir / f"backup_{timestamp}.tar.gz"
        
        import tarfile
        
        with tarfile.open(backup_path, 'w:gz') as tar:
            tar.add(self.memory_dir, arcname='memory')
        
        logger.info(f"Memory backup created: {backup_path}")
        return str(backup_path)
    
    def restore_memory(self, backup_path: str):
        """
        Restore memory from a backup.
        
        Args:
            backup_path (str): Path to the backup file.
        """
        import tarfile
        import shutil
        
        # Create temporary directory
        temp_dir = self.memory_dir.parent / "temp_restore"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract backup
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(temp_dir)
            
            # Replace current memory
            shutil.rmtree(self.memory_dir)
            shutil.move(temp_dir / "memory", self.memory_dir)
            
            # Reinitialize
            self._init_database()
            
            logger.info(f"Memory restored from backup: {backup_path}")
            
        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


def main():
    """
    Main function for testing the Memory Manager.
    """
    # Initialize memory manager
    memory = MemoryManager("test_memory")
    
    # Test data storage
    test_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003'],
        'outcome': [85, 78, 92],
        'compliance': [90, 85, 95]
    })
    
    data_id = memory.store_data('test_trial_data', test_data, 
                               metadata={'study': 'TEST001'}, 
                               tags=['trial', 'test'])
    
    print(f"Data stored with ID: {data_id}")
    
    # Test data retrieval
    retrieved_data = memory.get_data('test_trial_data')
    print(f"Retrieved data shape: {retrieved_data.shape}")
    
    # Test pattern storage
    pattern_id = memory.store_pattern(
        'test_pattern_001',
        'outcome_improvement',
        {'condition': 'compliance > 90', 'outcome': 'improved'},
        frequency=5,
        confidence=0.8
    )
    
    print(f"Pattern stored with ID: {pattern_id}")
    
    # Test pattern retrieval
    patterns = memory.get_patterns(pattern_type='outcome_improvement')
    print(f"Found {len(patterns)} patterns")
    
    # Test memory statistics
    stats = memory.get_memory_stats()
    print(f"Memory statistics: {stats}")
    
    # Test cleanup
    memory.cleanup_memory(force=True)
    print("Memory cleanup completed")


if __name__ == "__main__":
    main()