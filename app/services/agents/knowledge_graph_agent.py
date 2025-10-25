"""
Knowledge Graph Agent
Connects issues across platforms and builds relationships
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import sklearn, fallback to simple implementations
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Simple fallback implementations
    class TfidfVectorizer:
        def __init__(self, max_features=1000, stop_words='english'):
            self.max_features = max_features
            self.stop_words = stop_words
        
        def fit_transform(self, texts):
            return [[1.0] * 10 for _ in texts]  # Mock matrix
    
    def cosine_similarity(matrix):
        return [[0.5] * len(matrix) for _ in matrix]  # Mock similarity
    
    class KMeans:
        def __init__(self, n_clusters, random_state=None):
            self.n_clusters = n_clusters
        def fit_predict(self, matrix):
            return [i % self.n_clusters for i in range(len(matrix))]

from app.services.agent_framework import BaseAgent, AgentTask, AgentResult, AgentStatus
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeGraphTask:
    """Knowledge graph agent task"""
    task_id: str
    agent_id: str
    data: Dict[str, Any]
    new_issues: List[Dict[str, Any]]
    similarity_threshold: float = 0.7
    max_clusters: int = 10
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class KnowledgeGraphResult:
    """Knowledge graph agent result"""
    task_id: str
    agent_id: str
    status: AgentStatus
    data: Dict[str, Any]
    relationships_created: int
    clusters_identified: int
    insights_generated: List[str]
    knowledge_graph_updates: Dict[str, Any]
    insights: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    execution_time: float = 0.0
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class KnowledgeGraphAgent(BaseAgent):
    """Intelligent knowledge graph agent for connecting issues"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("knowledge_graph_agent", config)
        self.llm_service = LLMService()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.similarity_cache = {}
        
        # Smart sampling parameters
        self.sample_size = 50  # Sample size for LLM analysis
        self.batch_size = 20   # Process in batches
        self.relationship_patterns = {}  # Cache learned patterns
        
    async def execute(self, task: KnowledgeGraphTask) -> KnowledgeGraphResult:
        """Execute knowledge graph intelligence task"""
        logger.info(f"Knowledge graph agent processing {len(task.new_issues)} new issues")
        
        try:
            # Step 1: Find relationships between issues
            relationships = await self._find_relationships(task.new_issues, task.similarity_threshold)
            logger.info(f"Found {len(relationships)} relationships")
            
            # Step 2: Cluster similar issues
            clusters = await self._cluster_issues(task.new_issues, task.max_clusters)
            logger.info(f"Identified {len(clusters)} clusters")
            
            # Step 3: Generate insights
            insights = await self._generate_insights(relationships, clusters, task.new_issues)
            logger.info(f"Generated {len(insights)} insights")
            
            # Step 4: Update knowledge graph
            kg_updates = await self._update_knowledge_graph(relationships, clusters, insights)
            
            return KnowledgeGraphResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=AgentStatus.SUCCESS,
                data={
                    "new_issues": task.new_issues,
                    "relationships": relationships,
                    "clusters": clusters,
                    "insights": insights,
                    "kg_updates": kg_updates
                },
                relationships_created=len(relationships),
                clusters_identified=len(clusters),
                insights_generated=insights,
                knowledge_graph_updates=kg_updates,
                insights=[
                    f"Created {len(relationships)} relationships between issues",
                    f"Identified {len(clusters)} issue clusters",
                    f"Generated {len(insights)} insights",
                    f"Most connected issue type: {self._get_most_connected_type(relationships)}"
                ]
            )
            
        except Exception as e:
            logger.error(f"Knowledge graph agent failed: {str(e)}")
            raise
    
    async def _find_relationships(self, issues: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """Find relationships between issues using LLM with smart sampling"""
        relationships = []
        
        if len(issues) < 2:
            return relationships
        
        # Smart sampling: Use LLM to analyze a sample and apply patterns to all data
        if len(issues) > self.sample_size:
            # Sample representative issues for LLM analysis
            sample_issues = self._smart_sample_issues(issues, self.sample_size)
            # Use LLM to learn relationship patterns from sample
            learned_patterns = await self._llm_learn_relationship_patterns(sample_issues)
            # Apply learned patterns to all issues
            relationships = await self._apply_learned_patterns(issues, learned_patterns, threshold)
        else:
            # Small dataset: Use LLM directly
            relationships = await self._llm_find_relationships_direct(issues, threshold)
        
        return relationships
    
    def _smart_sample_issues(self, issues: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """Smart sampling to get representative issues for LLM analysis"""
        # Prioritize diverse issues by source, severity, and location
        source_groups = {}
        severity_groups = {}
        location_groups = {}
        
        for issue in issues:
            source = issue.get('source', 'unknown')
            severity = issue.get('severity', 'low')
            location = issue.get('location', {}).get('neighborhood', 'unknown')
            
            if source not in source_groups:
                source_groups[source] = []
            if severity not in severity_groups:
                severity_groups[severity] = []
            if location not in location_groups:
                location_groups[location] = []
                
            source_groups[source].append(issue)
            severity_groups[severity].append(issue)
            location_groups[location].append(issue)
        
        # Sample from each group proportionally
        sample = []
        for group in [source_groups, severity_groups, location_groups]:
            for items in group.values():
                if items:
                    # Take up to 2 items from each group
                    sample.extend(items[:2])
        
        # If we need more, add random samples
        if len(sample) < sample_size:
            remaining = [issue for issue in issues if issue not in sample]
            sample.extend(remaining[:sample_size - len(sample)])
        
        return sample[:sample_size]
    
    async def _llm_learn_relationship_patterns(self, sample_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to learn relationship patterns from a sample of issues"""
        
        # Prepare sample data for LLM
        issues_text = []
        for i, issue in enumerate(sample_issues):
            issues_text.append(f"Issue {i+1}: {issue.get('title', '')} | {issue.get('description', '')} | Source: {issue.get('source', '')} | Severity: {issue.get('severity', '')} | Location: {issue.get('location', {}).get('neighborhood', 'Unknown')}")
        
        issues_content = "\n".join(issues_text)
        
        prompt = f"""You are an expert urban planning analyst. Analyze these community issues and identify relationship patterns that can be applied to larger datasets.

Community Issues:
{issues_content}

Identify 3-5 key relationship patterns that can be used to connect similar issues. For each pattern, provide:

1. **Pattern Name**: Clear description of the relationship type
2. **Detection Criteria**: Specific rules to identify this relationship
3. **Confidence Score**: How reliable this pattern is (0-1)
4. **Examples**: Which issues in the sample demonstrate this pattern

Return ONLY a valid JSON object:
{{
  "patterns": [
    {{
      "name": "Geographic Co-location",
      "criteria": {{
        "location_match": true,
        "distance_threshold": 0.5,
        "neighborhood_same": true
      }},
      "confidence": 0.9,
      "examples": ["Issue 1", "Issue 3"]
    }},
    {{
      "name": "Thematic Similarity",
      "criteria": {{
        "keyword_overlap": ["homeless", "housing", "safety"],
        "severity_similar": true,
        "source_diverse": true
      }},
      "confidence": 0.8,
      "examples": ["Issue 2", "Issue 4"]
    }}
  ],
  "general_rules": {{
    "high_confidence_threshold": 0.8,
    "medium_confidence_threshold": 0.6,
    "relationship_types": ["geographic", "thematic", "temporal", "severity"]
  }}
}}

Focus on patterns that can be applied algorithmically to large datasets."""

        try:
            response = self.llm_service.provider.client.messages.create(
                model=self.llm_service.provider.model,
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                patterns = json.loads(json_match.group(0))
                logger.info(f"LLM learned {len(patterns.get('patterns', []))} relationship patterns")
                return patterns
            else:
                logger.warning("No valid JSON found in LLM response")
                return {"patterns": [], "general_rules": {}}
                
        except Exception as e:
            logger.error(f"LLM pattern learning failed: {str(e)}")
            return {"patterns": [], "general_rules": {}}
    
    async def _apply_learned_patterns(self, all_issues: List[Dict[str, Any]], patterns: Dict[str, Any], threshold: float) -> List[Dict[str, Any]]:
        """Apply learned patterns to all issues efficiently"""
        relationships = []
        learned_patterns = patterns.get('patterns', [])
        
        # Process in batches to avoid memory issues
        for i in range(0, len(all_issues), self.batch_size):
            batch = all_issues[i:i + self.batch_size]
            batch_relationships = await self._apply_patterns_to_batch(batch, learned_patterns, threshold)
            relationships.extend(batch_relationships)
        
        return relationships
    
    async def _apply_patterns_to_batch(self, batch: List[Dict[str, Any]], patterns: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """Apply patterns to a batch of issues"""
        relationships = []
        
        for pattern in patterns:
            pattern_name = pattern.get('name', '')
            criteria = pattern.get('criteria', {})
            confidence = pattern.get('confidence', 0.5)
            
            if confidence < threshold:
                continue
            
            # Apply pattern-specific logic
            if 'Geographic' in pattern_name:
                relationships.extend(self._apply_geographic_pattern(batch, criteria))
            elif 'Thematic' in pattern_name:
                relationships.extend(self._apply_thematic_pattern(batch, criteria))
            elif 'Temporal' in pattern_name:
                relationships.extend(self._apply_temporal_pattern(batch, criteria))
        
        return relationships
    
    def _apply_geographic_pattern(self, issues: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply geographic relationship pattern"""
        relationships = []
        
        for i, issue1 in enumerate(issues):
            for j, issue2 in enumerate(issues[i+1:], i+1):
                if self._are_geographically_related(issue1, issue2):
                    relationships.append({
                        "issue_1": issue1,
                        "issue_2": issue2,
                        "relationship_type": "geographic",
                        "strength": 0.8,
                        "pattern": "geographic_co_location"
                    })
        
        return relationships
    
    def _apply_thematic_pattern(self, issues: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply thematic similarity pattern"""
        relationships = []
        keywords = criteria.get('keyword_overlap', [])
        
        for i, issue1 in enumerate(issues):
            for j, issue2 in enumerate(issues[i+1:], i+1):
                if self._are_thematically_similar(issue1, issue2, keywords):
                    relationships.append({
                        "issue_1": issue1,
                        "issue_2": issue2,
                        "relationship_type": "similar",
                        "strength": 0.7,
                        "pattern": "thematic_similarity"
                    })
        
        return relationships
    
    def _apply_temporal_pattern(self, issues: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply temporal relationship pattern"""
        relationships = []
        
        for i, issue1 in enumerate(issues):
            for j, issue2 in enumerate(issues[i+1:], i+1):
                if self._are_temporally_related(issue1, issue2):
                    relationships.append({
                        "issue_1": issue1,
                        "issue_2": issue2,
                        "relationship_type": "temporal",
                        "strength": 0.6,
                        "pattern": "temporal_proximity"
                    })
        
        return relationships
    
    async def _llm_find_relationships_direct(self, issues: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """Use LLM directly for small datasets"""
        if len(issues) < 2:
            return []
        
        # Prepare issues for LLM analysis
        issues_text = []
        for i, issue in enumerate(issues):
            issues_text.append(f"Issue {i+1}: {issue.get('title', '')} | {issue.get('description', '')} | Source: {issue.get('source', '')} | Severity: {issue.get('severity', '')}")
        
        issues_content = "\n".join(issues_text)
        
        prompt = f"""Analyze these community issues and identify relationships between them.

Issues:
{issues_content}

Identify pairs of issues that are related. For each relationship, provide:

1. **Issue Pair**: Which issues are related
2. **Relationship Type**: geographic, thematic, temporal, or similar
3. **Strength**: How strong the relationship is (0-1)
4. **Reason**: Why these issues are related

Return ONLY a valid JSON array:
[
  {{
    "issue_1_index": 0,
    "issue_2_index": 2,
    "relationship_type": "geographic",
    "strength": 0.8,
    "reason": "Both issues occur in the same neighborhood"
  }}
]

Only include relationships with strength >= {threshold}."""

        try:
            response = self.llm_service.provider.client.messages.create(
                model=self.llm_service.provider.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            import json
            import re
            
            # Extract JSON array
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                relationships_data = json.loads(json_match.group(0))
                relationships = []
                
                for rel in relationships_data:
                    i1 = rel.get('issue_1_index', 0)
                    i2 = rel.get('issue_2_index', 0)
                    
                    if 0 <= i1 < len(issues) and 0 <= i2 < len(issues):
                        relationships.append({
                            "issue_1": issues[i1],
                            "issue_2": issues[i2],
                            "relationship_type": rel.get('relationship_type', 'similar'),
                            "strength": rel.get('strength', 0.5),
                            "reason": rel.get('reason', '')
                        })
                
                logger.info(f"LLM identified {len(relationships)} relationships directly")
                return relationships
            else:
                logger.warning("No valid JSON found in LLM response")
                return []
                
        except Exception as e:
            logger.error(f"LLM direct relationship analysis failed: {str(e)}")
            return []
    
    def _classify_relationship(self, issue1: Dict[str, Any], issue2: Dict[str, Any], similarity: float) -> str:
        """Classify the type of relationship between issues"""
        # Geographic relationship
        if self._are_geographically_related(issue1, issue2):
            return "geographic"
        
        # Temporal relationship
        if self._are_temporally_related(issue1, issue2):
            return "temporal"
        
        # Similar content
        if similarity > 0.8:
            return "similar"
        
        # Related content
        return "related"
    
    def _are_geographically_related(self, issue1: Dict[str, Any], issue2: Dict[str, Any]) -> bool:
        """Check if issues are geographically related"""
        loc1 = issue1.get('location', {})
        loc2 = issue2.get('location', {})
        
        # Same neighborhood
        if (loc1.get('neighborhood') and loc2.get('neighborhood') and 
            loc1.get('neighborhood') == loc2.get('neighborhood')):
            return True
        
        # Same city
        if (loc1.get('city') and loc2.get('city') and 
            loc1.get('city') == loc2.get('city')):
            return True
        
        return False
    
    def _are_temporally_related(self, issue1: Dict[str, Any], issue2: Dict[str, Any]) -> bool:
        """Check if issues are temporally related"""
        # This would be enhanced with actual timestamp data
        # For now, assume issues from same source are temporally related
        return issue1.get('source') == issue2.get('source')
    
    async def _cluster_issues(self, issues: List[Dict[str, Any]], max_clusters: int) -> List[Dict[str, Any]]:
        """Cluster similar issues"""
        if len(issues) < 2:
            return []
        
        # Prepare text data
        texts = []
        for issue in issues:
            text = f"{issue.get('title', '')} {issue.get('description', '')}"
            texts.append(text)
        
        try:
            # Vectorize texts
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Determine optimal number of clusters
            n_clusters = min(max_clusters, len(issues) // 2, 5)
            if n_clusters < 2:
                return []
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Group issues by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(issues[i])
            
            # Create cluster summaries
            cluster_summaries = []
            for cluster_id, cluster_issues in clusters.items():
                if len(cluster_issues) > 1:  # Only clusters with multiple issues
                    representative = self._find_representative_issue(cluster_issues)
                    cluster_name = self._generate_cluster_name(representative, cluster_issues)
                    
                    summary = {
                        "cluster_id": cluster_id,
                        "name": cluster_name,
                        "issue_count": len(cluster_issues),
                        "representative_issue": representative,
                        "common_themes": self._extract_common_themes(cluster_issues),
                        "geographic_distribution": self._analyze_geographic_distribution(cluster_issues),
                        "severity_distribution": self._analyze_severity_distribution(cluster_issues),
                        "issues": cluster_issues
                    }
                    cluster_summaries.append(summary)
            
            return cluster_summaries
        
        except Exception as e:
            logger.warning(f"Clustering failed: {str(e)}")
            return []
    
    def _find_representative_issue(self, cluster_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find representative issue for cluster"""
        # Simple heuristic: issue with highest severity or most detailed description
        representative = cluster_issues[0]
        
        for issue in cluster_issues:
            if (issue.get('severity') == 'high' and representative.get('severity') != 'high'):
                representative = issue
            elif (len(issue.get('description', '')) > len(representative.get('description', ''))):
                representative = issue
        
        return representative
    
    def _generate_cluster_name(self, representative: Dict[str, Any], cluster_issues: List[Dict[str, Any]]) -> str:
        """Generate a meaningful name for the cluster"""
        # Extract key themes from the representative issue
        title = representative.get('title', '').lower()
        description = representative.get('description', '').lower()
        
        # Common issue patterns
        if any(word in title + description for word in ['homeless', 'housing', 'shelter']):
            return "Housing & Homelessness Issues"
        elif any(word in title + description for word in ['graffiti', 'vandalism', 'tag']):
            return "Graffiti & Vandalism"
        elif any(word in title + description for word in ['street', 'sidewalk', 'cleaning']):
            return "Street Maintenance"
        elif any(word in title + description for word in ['safety', 'crime', 'security']):
            return "Public Safety"
        elif any(word in title + description for word in ['traffic', 'parking', 'transit']):
            return "Transportation"
        else:
            # Use the first few words of the representative issue title
            words = title.split()[:3]
            return " ".join(word.capitalize() for word in words if word) or "Community Issues"
    
    def _extract_common_themes(self, cluster_issues: List[Dict[str, Any]]) -> List[str]:
        """Extract common themes from cluster"""
        # Simple keyword extraction
        all_text = ' '.join([issue.get('title', '') + ' ' + issue.get('description', '') 
                            for issue in cluster_issues])
        
        # Common SF issue keywords
        issue_keywords = [
            'homelessness', 'housing', 'crime', 'safety', 'traffic',
            'parking', 'transit', 'bart', 'graffiti', 'litter',
            'rent', 'cost of living', 'infrastructure', 'homeless',
            'encampment', 'dumping', 'noise', 'light', 'pothole'
        ]
        
        themes = []
        for keyword in issue_keywords:
            if keyword.lower() in all_text.lower():
                themes.append(keyword)
        
        return themes[:5]  # Top 5 themes
    
    def _analyze_geographic_distribution(self, cluster_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze geographic distribution of cluster"""
        neighborhoods = {}
        cities = {}
        
        for issue in cluster_issues:
            location = issue.get('location', {})
            neighborhood = location.get('neighborhood')
            city = location.get('city')
            
            if neighborhood:
                neighborhoods[neighborhood] = neighborhoods.get(neighborhood, 0) + 1
            if city:
                cities[city] = cities.get(city, 0) + 1
        
        return {
            "neighborhoods": neighborhoods,
            "cities": cities,
            "most_common_neighborhood": max(neighborhoods.keys(), key=neighborhoods.get) if neighborhoods else None,
            "most_common_city": max(cities.keys(), key=cities.get) if cities else None
        }
    
    def _analyze_severity_distribution(self, cluster_issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze severity distribution of cluster"""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for issue in cluster_issues:
            severity = issue.get('severity', 'low')
            distribution[severity] += 1
        
        return distribution
    
    async def _generate_insights(self, relationships: List[Dict[str, Any]], 
                               clusters: List[Dict[str, Any]], 
                               issues: List[Dict[str, Any]]) -> List[str]:
        """Generate meaningful insights from analysis results"""
        insights = []
        
        # Basic statistics
        insights.append(f"Analyzed {len(issues)} community issues")
        
        # Relationship insights
        if relationships:
            insights.append(f"Discovered {len(relationships)} issue relationships")
            
            # Most common relationship type
            relationship_types = {}
            for rel in relationships:
                rel_type = rel.get('relationship_type', 'unknown')
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
            if relationship_types:
                most_common = max(relationship_types.keys(), key=relationship_types.get)
                insights.append(f"Primary connection type: {most_common} relationships")
        
        # Cluster insights
        if clusters:
            insights.append(f"Organized issues into {len(clusters)} thematic clusters")
            
            # Cluster details
            for cluster in clusters:
                cluster_name = cluster.get('name', 'Unknown Cluster')
                issue_count = cluster.get('issue_count', 0)
                insights.append(f"• {cluster_name}: {issue_count} related issues")
        
        # Source diversity
        sources = {}
        for issue in issues:
            source = issue.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        if len(sources) > 1:
            source_summary = ", ".join([f"{count} from {source}" for source, count in sources.items()])
            insights.append(f"Data sources: {source_summary}")
        
        # Geographic insights
        neighborhoods = {}
        for issue in issues:
            location = issue.get('location', {})
            neighborhood = location.get('neighborhood')
            if neighborhood:
                neighborhoods[neighborhood] = neighborhoods.get(neighborhood, 0) + 1
        
        if neighborhoods:
            most_active = max(neighborhoods.keys(), key=neighborhoods.get)
            insights.append(f"Most affected area: {most_active} ({neighborhoods[most_active]} issues)")
        
        # Severity analysis
        severities = {}
        for issue in issues:
            severity = issue.get('severity', 'unknown')
            severities[severity] = severities.get(severity, 0) + 1
        
        if severities:
            high_severity = severities.get('high', 0)
            if high_severity > 0:
                insights.append(f"Critical issues identified: {high_severity} high-severity problems")
        
        return insights
    
    async def _update_knowledge_graph(self, relationships: List[Dict[str, Any]], 
                                    clusters: List[Dict[str, Any]], 
                                    insights: List[str]) -> Dict[str, Any]:
        """Update knowledge graph with new information"""
        return {
            "relationships_added": len(relationships),
            "clusters_created": len(clusters),
            "insights_generated": len(insights),
            "timestamp": datetime.utcnow().isoformat(),
            "graph_statistics": {
                "total_relationships": len(relationships),
                "total_clusters": len(clusters),
                "relationship_types": self._count_relationship_types(relationships),
                "cluster_sizes": [c['issue_count'] for c in clusters]
            }
        }
    
    def _count_relationship_types(self, relationships: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count relationship types"""
        types = {}
        for relationship in relationships:
            rel_type = relationship.get('relationship_type', 'unknown')
            types[rel_type] = types.get(rel_type, 0) + 1
        return types
    
    def _get_most_connected_type(self, relationships: List[Dict[str, Any]]) -> str:
        """Get most connected issue type"""
        if not relationships:
            return "None"
        
        # Count issue types in relationships
        type_counts = {}
        for relationship in relationships:
            issue1 = relationship.get('issue_1', {})
            issue2 = relationship.get('issue_2', {})
            
            # Extract issue types from titles or descriptions
            type1 = self._extract_issue_type(issue1)
            type2 = self._extract_issue_type(issue2)
            
            if type1:
                type_counts[type1] = type_counts.get(type1, 0) + 1
            if type2:
                type_counts[type2] = type_counts.get(type2, 0) + 1
        
        if not type_counts:
            return "Unknown"
        
        return max(type_counts.keys(), key=type_counts.get)
    
    def _extract_issue_type(self, issue: Dict[str, Any]) -> Optional[str]:
        """Extract issue type from issue data"""
        title = issue.get('title', '').lower()
        description = issue.get('description', '').lower()
        
        # Common issue types
        issue_types = [
            'graffiti', 'homelessness', 'traffic', 'parking', 'litter',
            'crime', 'safety', 'housing', 'transit', 'infrastructure'
        ]
        
        for issue_type in issue_types:
            if issue_type in title or issue_type in description:
                return issue_type
        
        return None
