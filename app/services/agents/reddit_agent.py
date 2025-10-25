"""
Reddit Intelligence Agent
Smart subreddit discovery and content analysis
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.services.agent_framework import BaseAgent, AgentTask, AgentResult, AgentStatus
from app.services.llm_service import LLMService
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../parsers'))

# Import Reddit parser functions
try:
    from parsers.redditParser import find_subreddits, scrape_subreddits
    SubredditFinder = find_subreddits.SubredditFinder
    SubredditScraper = scrape_subreddits.SubredditScraper
except ImportError:
    # Fallback - create mock classes for testing
    class SubredditFinder:
        def find_subreddits(self, **kwargs):
            return {"subreddit_names": ["sftransportation", "sanfrancisco"]}
    
    class SubredditScraper:
        def scrape_subreddits(self, **kwargs):
            return {"total_posts": 10, "issue_posts": []}

logger = logging.getLogger(__name__)

@dataclass
class RedditTask:
    """Reddit agent task"""
    task_id: str
    agent_id: str
    data: Dict[str, Any]
    city: str
    keywords: List[str]
    max_subreddits: int = 10
    max_posts_per_subreddit: int = 25
    issue_keywords: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class RedditResult:
    """Reddit agent result"""
    task_id: str
    agent_id: str
    status: AgentStatus
    data: Dict[str, Any]
    subreddits_found: List[str]
    posts_analyzed: int
    issues_identified: List[Dict[str, Any]]
    geographic_insights: Dict[str, Any]
    insights: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    execution_time: float = 0.0
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class RedditAgent(BaseAgent):
    """Intelligent Reddit agent for city issues"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("reddit_agent", config)
        self.finder = SubredditFinder()
        self.scraper = SubredditScraper()
        self.llm_service = LLMService()
        
        # Default issue keywords for San Francisco
        self.default_issue_keywords = [
            'homelessness', 'housing crisis', 'crime', 'public safety',
            'BART', 'traffic', 'cost of living', 'rent', 'graffiti',
            'litter', 'parking', 'transit', 'infrastructure'
        ]
    
    async def execute(self, task: RedditTask) -> RedditResult:
        """Execute Reddit intelligence task"""
        logger.info(f"Reddit agent starting for city: {task.city}")
        
        try:
            # Step 1: Adaptive subreddit discovery
            subreddits = await self._discover_subreddits(task)
            logger.info(f"Found {len(subreddits)} relevant subreddits")
            
            # Step 2: Smart content scraping
            posts_data = await self._scrape_content(task, subreddits)
            logger.info(f"Scraped {posts_data.get('stats', {}).get('total_posts', 0)} posts")
            
            # Step 3: Issue identification and geographic mapping
            issues = await self._identify_issues(posts_data, task)
            logger.info(f"Identified {len(issues)} community issues")
            
            # Step 4: Generate insights
            insights = await self._generate_insights(issues, posts_data)
            
            return RedditResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=AgentStatus.SUCCESS,
                data={
                    "city": task.city,
                    "subreddits": subreddits,
                    "posts_data": posts_data,
                    "issues": issues
                },
                subreddits_found=subreddits,
                posts_analyzed=posts_data.get('stats', {}).get('total_posts', 0),
                issues_identified=issues,
                geographic_insights=insights,
                insights=[
                    f"Found {len(subreddits)} relevant subreddits for {task.city}",
                    f"Identified {len(issues)} community issues",
                    f"Most active subreddit: {self._get_most_active_subreddit(posts_data)}"
                ]
            )
            
        except Exception as e:
            logger.error(f"Reddit agent failed: {str(e)}")
            raise
    
    async def _discover_subreddits(self, task: RedditTask) -> List[str]:
        """Rate-limit optimized subreddit discovery"""
        # Use LLM to generate the most effective search strategy
        search_strategy = await self._llm_optimize_search_strategy(task)
        
        # Use only the most effective keywords to minimize API calls
        optimized_keywords = search_strategy['keywords'][:3]  # Limit to top 3 keywords
        
        # Use conservative search parameters to avoid rate limits
        search_params = {
            'min_subscribers': 5000,  # Higher threshold to get quality subreddits
            'min_relevance_score': 30.0,  # Higher threshold for relevance
            'max_results': min(task.max_subreddits, 3)  # Limit results
        }
        
        # Single optimized search call
        discovery_results = self.finder.find_subreddits(
            city=task.city,
            keywords=optimized_keywords,
            min_subscribers=search_params['min_subscribers'],
            max_results=search_params['max_results'],
            min_relevance_score=search_params['min_relevance_score']
        )
        
        return discovery_results['subreddit_names']
    
    async def _scrape_content(self, task: RedditTask, subreddits: List[str]) -> Dict[str, Any]:
        """Rate-limit optimized content scraping"""
        # Use LLM to optimize scraping strategy
        scraping_strategy = await self._llm_optimize_scraping_strategy(task, subreddits)
        
        # Conservative scraping parameters to respect rate limits
        scraping_params = {
            'get_recent': True,  # Only get recent posts
            'get_weekly_top': False,  # Skip weekly top to save API calls
            'search_issues': True,  # Use targeted search
            'max_posts_per_subreddit': min(task.max_posts_per_subreddit, 10)  # Limit posts
        }
        
        # Use optimized keywords from LLM
        optimized_keywords = scraping_strategy['keywords'][:5]  # Limit to top 5 keywords
        
        scraping_results = self.scraper.scrape_subreddits(
            subreddit_names=subreddits,
            issue_keywords=optimized_keywords,
            get_recent=scraping_params['get_recent'],
            get_weekly_top=scraping_params['get_weekly_top'],
            search_issues=scraping_params['search_issues']
        )
        
        return scraping_results
    
    async def _llm_optimize_search_strategy(self, task: RedditTask) -> Dict[str, Any]:
        """Use LLM to optimize search strategy for minimal API calls"""
        prompt = f"""You are a Reddit API optimization expert. Given these search parameters for {task.city}, generate the most efficient search strategy that maximizes results while minimizing API calls.

Original keywords: {task.keywords}
City: {task.city}

Generate a strategy that:
1. Uses only the 3 most effective keywords
2. Focuses on high-impact, specific terms
3. Avoids generic terms that return too many results
4. Prioritizes terms likely to find relevant community issues

Return ONLY a JSON object:
{{
  "keywords": ["most_effective_keyword1", "most_effective_keyword2", "most_effective_keyword3"],
  "strategy": "brief explanation of why these keywords are optimal",
  "expected_results": "estimated number of relevant subreddits"
}}"""

        try:
            response = self.llm_service.provider.client.messages.create(
                model=self.llm_service.provider.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            import json
            import re
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # Fallback to original keywords
                return {"keywords": task.keywords[:3], "strategy": "fallback", "expected_results": "unknown"}
                
        except Exception as e:
            logger.warning(f"LLM search optimization failed: {str(e)}")
            return {"keywords": task.keywords[:3], "strategy": "fallback", "expected_results": "unknown"}
    
    async def _llm_optimize_scraping_strategy(self, task: RedditTask, subreddits: List[str]) -> Dict[str, Any]:
        """Use LLM to optimize scraping strategy for minimal API calls"""
        prompt = f"""You are a Reddit scraping optimization expert. Given these subreddits and keywords, generate the most efficient scraping strategy.

Subreddits: {subreddits}
Original keywords: {task.keywords}
City: {task.city}

Generate a strategy that:
1. Uses only the 5 most effective issue keywords
2. Focuses on terms likely to find community problems
3. Avoids overly broad terms
4. Prioritizes actionable issues

Return ONLY a JSON object:
{{
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "strategy": "brief explanation of optimization approach",
  "expected_posts": "estimated number of relevant posts"
}}"""

        try:
            response = self.llm_service.provider.client.messages.create(
                model=self.llm_service.provider.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            import json
            import re
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # Fallback to original keywords
                return {"keywords": task.keywords[:5], "strategy": "fallback", "expected_posts": "unknown"}
                
        except Exception as e:
            logger.warning(f"LLM scraping optimization failed: {str(e)}")
            return {"keywords": task.keywords[:5], "strategy": "fallback", "expected_posts": "unknown"}
    
    async def _identify_issues(self, posts_data: Dict[str, Any], task: RedditTask) -> List[Dict[str, Any]]:
        """Identify community issues from posts using LLM analysis"""
        issues = []
        
        # Process issue posts
        if 'issue_posts' in posts_data:
            # Use LLM to analyze posts and extract structured issues
            posts_text = []
            for post in posts_data['issue_posts'][:10]:  # Limit to top 10 for LLM analysis
                posts_text.append(f"Title: {post.get('title', '')}\nContent: {post.get('selftext', '')}\nScore: {post.get('score', 0)}")
            
            if posts_text:
                # Use LLM to analyze and structure the issues
                analyzed_issues = await self._llm_analyze_reddit_issues(posts_text, task.city)
                issues.extend(analyzed_issues)
        
        return issues
    
    async def _llm_analyze_reddit_issues(self, posts_text: List[str], city: str) -> List[Dict[str, Any]]:
        """Use LLM to analyze Reddit posts and extract structured community issues"""
        posts_content = "\n\n---\n\n".join(posts_text)
        
        prompt = f"""You are an expert urban planning analyst. Analyze these Reddit posts from {city} residents and identify the most pressing community issues.

Posts from residents:
{posts_content}

Extract 2-4 most significant community issues. For each issue, provide:

1. **Issue Title**: Concise problem description (5-8 words)
2. **Description**: Detailed explanation with specific examples from the posts
3. **Severity**: high/medium/low based on frequency and impact
4. **Location**: Specific neighborhood or area if mentioned
5. **Evidence**: Direct quotes from posts that support this issue

Return ONLY a valid JSON array:
[
  {{
    "title": "Issue Title",
    "description": "Detailed description with specific examples from posts",
    "severity": "high|medium|low",
    "source": "reddit",
    "source_id": "reddit_analysis",
    "location": {{
      "city": "{city}",
      "neighborhood": "specific neighborhood if mentioned",
      "coordinates": null
    }},
    "metadata": {{
      "analysis_type": "llm_reddit_analysis",
      "evidence_posts": ["quote1", "quote2"],
      "confidence": 0.8
    }}
  }}
]

Focus on issues that:
- Are mentioned multiple times
- Have high community impact
- Are actionable by local government
- Include specific examples from the posts"""

        try:
            # Use the LLM service to analyze the posts
            response = self.llm_service.provider.client.messages.create(
                model=self.llm_service.provider.model,
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            # Extract JSON from response
            import json
            import re
            
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                issues = json.loads(json_match.group(0))
                logger.info(f"LLM identified {len(issues)} community issues from Reddit posts")
                return issues
            else:
                logger.warning("No valid JSON found in LLM response")
                return []
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            # Fallback to simple analysis
            return self._fallback_issue_analysis(posts_text, city)
    
    def _fallback_issue_analysis(self, posts_text: List[str], city: str) -> List[Dict[str, Any]]:
        """Fallback simple issue analysis without LLM"""
        issues = []
        
        # Simple keyword-based analysis
        issue_keywords = {
            'homelessness': ['homeless', 'encampment', 'tent', 'shelter'],
            'transportation': ['bart', 'muni', 'traffic', 'delays', 'transit'],
            'housing': ['rent', 'eviction', 'affordable', 'cost of living'],
            'safety': ['crime', 'unsafe', 'dangerous', 'police']
        }
        
        for category, keywords in issue_keywords.items():
            mentions = 0
            evidence = []
            for post in posts_text:
                for keyword in keywords:
                    if keyword.lower() in post.lower():
                        mentions += 1
                        evidence.append(post[:100] + "...")
            
            if mentions > 0:
                issues.append({
                    "title": f"{category.title()} Issues",
                    "description": f"Multiple mentions of {category} issues in community discussions",
                    "severity": "high" if mentions > 3 else "medium",
                    "source": "reddit",
                    "source_id": f"reddit_{category}",
                    "location": {"city": city, "neighborhood": None, "coordinates": None},
                    "metadata": {
                        "analysis_type": "keyword_fallback",
                        "evidence_posts": evidence[:3],
                        "confidence": 0.6
                    }
                })
        
        return issues
    
    async def _generate_insights(self, issues: List[Dict[str, Any]], posts_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate geographic and thematic insights"""
        insights = {
            "total_issues": len(issues),
            "severity_distribution": self._calculate_severity_distribution(issues),
            "top_subreddits": self._get_top_subreddits(posts_data),
            "geographic_patterns": self._analyze_geographic_patterns(issues),
            "trending_topics": self._identify_trending_topics(issues)
        }
        
        return insights
    
    def _evolve_keywords(self, base_keywords: List[str]) -> List[str]:
        """Evolve keywords based on successful patterns"""
        # Get successful patterns from memory
        successful_patterns = self.memory.get_successful_patterns()
        
        # Start with base keywords
        evolved = base_keywords.copy()
        
        # Add successful keywords from memory
        for pattern in successful_patterns:
            if 'keywords' in pattern.get('pattern', {}):
                evolved.extend(pattern['pattern']['keywords'])
        
        # Add city-specific terms
        evolved.extend(['sf', 'san francisco', 'bay area'])
        
        # Remove duplicates and return
        return list(set(evolved))
    
    def _assess_severity(self, post: Dict[str, Any]) -> str:
        """Assess issue severity based on post content"""
        title = post.get('title', '').lower()
        content = post.get('selftext', '').lower()
        score = post.get('score', 0)
        
        # High severity indicators
        high_severity_terms = ['emergency', 'urgent', 'dangerous', 'unsafe', 'crisis']
        if any(term in title or term in content for term in high_severity_terms):
            return 'high'
        
        # Medium severity indicators
        medium_severity_terms = ['problem', 'issue', 'concern', 'complaint']
        if any(term in title or term in content for term in medium_severity_terms):
            return 'medium'
        
        # Low severity by default
        return 'low'
    
    def _extract_location(self, post: Dict[str, Any], city: str) -> Dict[str, Any]:
        """Extract location information from post"""
        title = post.get('title', '')
        content = post.get('selftext', '')
        
        # Simple location extraction (can be enhanced with NLP)
        location = {
            "city": city,
            "neighborhood": None,
            "coordinates": None,
            "address": None
        }
        
        # Look for neighborhood mentions
        sf_neighborhoods = [
            'mission', 'castro', 'haight', 'richmond', 'sunset',
            'marina', 'pacific heights', 'nob hill', 'chinatown',
            'north beach', 'soma', 'tenderloin', 'hayes valley'
        ]
        
        text = (title + ' ' + content).lower()
        for neighborhood in sf_neighborhoods:
            if neighborhood in text:
                location["neighborhood"] = neighborhood.title()
                break
        
        return location
    
    def _calculate_severity_distribution(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate severity distribution"""
        distribution = {"high": 0, "medium": 0, "low": 0}
        for issue in issues:
            severity = issue.get('severity', 'low')
            distribution[severity] += 1
        return distribution
    
    def _get_top_subreddits(self, posts_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get top subreddits by activity"""
        if 'subreddit_stats' not in posts_data:
            return []
        
        stats = posts_data['subreddit_stats']
        return sorted(stats, key=lambda x: x.get('post_count', 0), reverse=True)[:5]
    
    def _analyze_geographic_patterns(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze geographic patterns in issues"""
        neighborhoods = {}
        for issue in issues:
            location = issue.get('location', {})
            neighborhood = location.get('neighborhood')
            if neighborhood:
                if neighborhood not in neighborhoods:
                    neighborhoods[neighborhood] = 0
                neighborhoods[neighborhood] += 1
        
        return {
            "neighborhood_counts": neighborhoods,
            "most_mentioned_neighborhood": max(neighborhoods.keys(), key=neighborhoods.get) if neighborhoods else None
        }
    
    def _identify_trending_topics(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Identify trending topics from issues"""
        # Simple keyword frequency analysis
        all_text = ' '.join([issue.get('title', '') + ' ' + issue.get('description', '') for issue in issues])
        
        # Common SF issue keywords
        issue_keywords = [
            'homelessness', 'housing', 'crime', 'safety', 'traffic',
            'parking', 'transit', 'bart', 'graffiti', 'litter',
            'rent', 'cost of living', 'infrastructure'
        ]
        
        trending = []
        for keyword in issue_keywords:
            if keyword.lower() in all_text.lower():
                trending.append(keyword)
        
        return trending[:5]  # Top 5 trending topics
    
    def _get_adaptive_search_params(self) -> Dict[str, Any]:
        """Get adaptive search parameters based on past success"""
        successful_patterns = self.memory.get_successful_patterns()
        
        # Default parameters
        params = {
            'min_subscribers': 1000,
            'min_relevance_score': 20.0
        }
        
        # Learn from successful patterns
        if successful_patterns:
            # If we've had success with lower thresholds, use them
            recent_success = successful_patterns[-1]
            if recent_success.get('pattern', {}).get('min_subscribers', 1000) < 1000:
                params['min_subscribers'] = max(500, recent_success['pattern']['min_subscribers'])
            
            if recent_success.get('pattern', {}).get('min_relevance_score', 20.0) < 20.0:
                params['min_relevance_score'] = max(10.0, recent_success['pattern']['min_relevance_score'])
        
        return params
    
    def _learn_from_discovery(self, discovery_results: Dict[str, Any], keywords: List[str]):
        """Learn from subreddit discovery results"""
        if discovery_results.get('subreddit_names'):
            # Record successful discovery pattern
            pattern = {
                'keywords': keywords,
                'min_subscribers': 1000,
                'min_relevance_score': 20.0,
                'subreddits_found': len(discovery_results['subreddit_names'])
            }
            
            # Create mock result for memory
            mock_result = type('MockResult', (), {
                'data': {'subreddits_found': len(discovery_results['subreddit_names'])},
                'status': 'success'
            })()
            
            self.memory.record_success(pattern, mock_result)
    
    def _evolve_issue_keywords(self, base_keywords: Optional[List[str]]) -> List[str]:
        """Evolve issue keywords based on successful patterns"""
        if base_keywords:
            evolved = base_keywords.copy()
        else:
            evolved = self.default_issue_keywords.copy()
        
        # Get successful patterns from memory
        successful_patterns = self.memory.get_successful_patterns()
        
        # Add successful issue keywords from memory
        for pattern in successful_patterns:
            if 'issue_keywords' in pattern.get('pattern', {}):
                evolved.extend(pattern['pattern']['issue_keywords'])
        
        # Add trending SF-specific terms
        sf_trending = ['homeless', 'encampment', 'tent', 'shelter', 'housing', 'rent', 'eviction']
        evolved.extend(sf_trending)
        
        # Remove duplicates and return
        return list(set(evolved))
    
    def _get_adaptive_scraping_params(self) -> Dict[str, Any]:
        """Get adaptive scraping parameters based on past success"""
        successful_patterns = self.memory.get_successful_patterns()
        
        # Default parameters
        params = {
            'sort': 'hot',
            'limit': 25,
            'time_filter': 'week'
        }
        
        # Learn from successful patterns
        if successful_patterns:
            recent_success = successful_patterns[-1]
            if 'scraping_params' in recent_success.get('pattern', {}):
                params.update(recent_success['pattern']['scraping_params'])
        
        return params
    
    def _learn_from_scraping(self, scraping_results: Dict[str, Any], keywords: List[str]):
        """Learn from scraping results"""
        if scraping_results.get('stats', {}).get('total_posts', 0) > 0:
            # Record successful scraping pattern
            pattern = {
                'issue_keywords': keywords,
                'scraping_params': {
                    'sort': 'hot',
                    'limit': 25,
                    'time_filter': 'week'
                },
                'posts_found': scraping_results.get('stats', {}).get('total_posts', 0)
            }
            
            # Create mock result for memory
            mock_result = type('MockResult', (), {
                'data': {'posts_found': scraping_results.get('stats', {}).get('total_posts', 0)},
                'status': 'success'
            })()
            
            self.memory.record_success(pattern, mock_result)
    
    def _get_most_active_subreddit(self, posts_data: Dict[str, Any]) -> str:
        """Get most active subreddit"""
        if 'subreddit_stats' not in posts_data or not posts_data['subreddit_stats']:
            return "Unknown"
        
        stats = posts_data['subreddit_stats']
        return max(stats, key=lambda x: x.get('post_count', 0)).get('subreddit', 'Unknown')
