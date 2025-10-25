from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.database import get_db, CityGoal, PolicyDocument, GoalRecommendation
from app.models import (
    CityGoalCreate, CityGoalResponse, PolicyDocumentCreate, PolicyDocumentResponse,
    TrainGoalsRequest, RAGQueryRequest, GoalRecommendationResponse
)
from app.services.vector_service import VectorService
from app.services.claude_service import ClaudeService
from typing import List, Optional
import json
from datetime import datetime

router = APIRouter()

# Initialize services
vector_service = VectorService()
claude_service = ClaudeService()

@router.post("/goals/train")
async def train_goals(
    request: TrainGoalsRequest,
    db: AsyncSession = Depends(get_db)
):
    """Store city goals and policy documents with embeddings"""
    
    try:
        print(f"🎯 Training goals for {request.city_name}")
        
        # Store goals in database
        stored_goals = []
        for goal_data in request.goals:
            # Create embedding for the goal
            goal_text = f"{goal_data.goal_title} {goal_data.goal_description} {goal_data.target_metric}"
            embedding = vector_service.create_embedding(goal_text)
            
            # Store in database
            goal = CityGoal(
                city_name=goal_data.city_name,
                goal_title=goal_data.goal_title,
                goal_description=goal_data.goal_description,
                target_metric=goal_data.target_metric,
                target_value=goal_data.target_value,
                target_unit=goal_data.target_unit,
                priority_level=goal_data.priority_level,
                deadline=goal_data.deadline,
                embedding_vector=json.dumps(embedding),
                metadata_json=goal_data.metadata_json
            )
            db.add(goal)
            await db.commit()
            await db.refresh(goal)
            stored_goals.append(goal)
        
        # Store policy documents if provided
        stored_policies = []
        for policy_data in request.policy_documents:
            # Create embedding for the policy
            policy_text = f"{policy_data.title} {policy_data.content}"
            embedding = vector_service.create_embedding(policy_text)
            
            # Store in database
            policy = PolicyDocument(
                source=policy_data.source,
                title=policy_data.title,
                content=policy_data.content,
                document_type=policy_data.document_type,
                geographic_scope=policy_data.geographic_scope,
                topic_tags=json.dumps(policy_data.topic_tags),
                embedding_vector=json.dumps(embedding)
            )
            db.add(policy)
            await db.commit()
            await db.refresh(policy)
            stored_policies.append(policy)
        
        # Add to vector index
        goal_dicts = [{
            'id': goal.id,
            'city_name': goal.city_name,
            'goal_title': goal.goal_title,
            'goal_description': goal.goal_description,
            'target_metric': goal.target_metric,
            'priority_level': goal.priority_level
        } for goal in stored_goals]
        
        policy_dicts = [{
            'id': policy.id,
            'source': policy.source,
            'title': policy.title,
            'content': policy.content,
            'document_type': policy.document_type,
            'geographic_scope': policy.geographic_scope,
            'topic_tags': json.loads(policy.topic_tags) if policy.topic_tags else []
        } for policy in stored_policies]
        
        # Add to vector index
        if goal_dicts:
            vector_service.add_goals_to_index(goal_dicts)
        if policy_dicts:
            vector_service.add_policies_to_index(policy_dicts)
        
        return {
            "success": True,
            "message": f"Successfully trained {len(stored_goals)} goals and {len(stored_policies)} policies for {request.city_name}",
            "goals_created": len(stored_goals),
            "policies_created": len(stored_policies),
            "vector_index_stats": vector_service.get_index_stats()
        }
        
    except Exception as e:
        print(f"❌ Error training goals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train goals: {str(e)}")

@router.get("/goals/{city_name}", response_model=List[CityGoalResponse])
async def get_city_goals(
    city_name: str,
    status: Optional[str] = Query(None, description="Filter by status: active, completed, paused"),
    db: AsyncSession = Depends(get_db)
):
    """Get all goals for a specific city"""
    
    query = select(CityGoal).where(CityGoal.city_name == city_name)
    if status:
        query = query.where(CityGoal.status == status)
    
    result = await db.execute(query)
    goals = result.scalars().all()
    
    return [CityGoalResponse(
        id=goal.id,
        city_name=goal.city_name,
        goal_title=goal.goal_title,
        goal_description=goal.goal_description,
        target_metric=goal.target_metric,
        target_value=goal.target_value,
        target_unit=goal.target_unit,
        priority_level=goal.priority_level,
        deadline=goal.deadline,
        status=goal.status,
        created_at=goal.created_at,
        updated_at=goal.updated_at
    ) for goal in goals]

@router.get("/goals/{city_name}/recommendations")
async def get_goal_recommendations(
    city_name: str,
    problem_description: str = Query(..., description="Description of the problem to solve"),
    current_data: str = Query(..., description="JSON string of current data context"),
    max_recommendations: int = Query(5, description="Maximum number of recommendations"),
    db: AsyncSession = Depends(get_db)
):
    """Get goal-aligned recommendations for a city's problems"""
    
    try:
        # Parse current data
        try:
            data_dict = json.loads(current_data)
        except json.JSONDecodeError:
            data_dict = {}
        
        # Get recommendations from vector service
        recommendations = vector_service.get_goal_aligned_recommendations(
            city_name=city_name,
            problem_description=problem_description,
            current_data=data_dict,
            max_recommendations=max_recommendations
        )
        
        # Enhance recommendations with LLM synthesis
        enhanced_recommendations = []
        for rec in recommendations:
            city_goal = rec['city_goal']
            policy_doc = rec['policy_document']
            
            # Create detailed recommendation using LLM
            llm_prompt = f"""
            City Goal: {city_goal['metadata'].get('original_text', '')}
            Policy Reference: {policy_doc['metadata'].get('original_text', '')}
            Problem: {problem_description}
            Current Data: {json.dumps(data_dict, default=str)}
            
            Create a specific, actionable recommendation that aligns the city's goal with the policy approach.
            Include implementation steps and expected impact.
            """
            
            try:
                # Use Claude to generate detailed recommendation
                llm_response = await claude_service.synthesize_problems(
                    {"recommendation_context": llm_prompt}, 
                    f"Goal-aligned recommendation for {city_name}"
                )
                
                enhanced_rec = {
                    'city_goal': city_goal,
                    'policy_document': policy_doc,
                    'similarity_score': rec['combined_score'],
                    'recommendation_text': llm_response[0].get('description', '') if llm_response else 'Goal-aligned recommendation available',
                    'implementation_steps': llm_response[0].get('solution', {}).get('description', '') if llm_response else '',
                    'estimated_impact': llm_response[0].get('solution', {}).get('expected_impact', '') if llm_response else ''
                }
                enhanced_recommendations.append(enhanced_rec)
                
            except Exception as e:
                print(f"⚠️ LLM enhancement failed: {e}")
                # Fallback to basic recommendation
                enhanced_rec = {
                    'city_goal': city_goal,
                    'policy_document': policy_doc,
                    'similarity_score': rec['combined_score'],
                    'recommendation_text': f"Aligns with goal: {city_goal['metadata'].get('original_text', '')}",
                    'implementation_steps': 'Review policy document for implementation details',
                    'estimated_impact': 'Expected to support city goal achievement'
                }
                enhanced_recommendations.append(enhanced_rec)
        
        return {
            "city_name": city_name,
            "problem_description": problem_description,
            "recommendations": enhanced_recommendations,
            "total_found": len(enhanced_recommendations)
        }
        
    except Exception as e:
        print(f"❌ Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@router.get("/goals/{city_name}/stats")
async def get_city_goals_stats(
    city_name: str,
    db: AsyncSession = Depends(get_db)
):
    """Get statistics about a city's goals and training status"""
    
    # Get goals count by status
    active_goals = await db.execute(
        select(CityGoal).where(and_(CityGoal.city_name == city_name, CityGoal.status == "active"))
    )
    active_count = len(active_goals.scalars().all())
    
    completed_goals = await db.execute(
        select(CityGoal).where(and_(CityGoal.city_name == city_name, CityGoal.status == "completed"))
    )
    completed_count = len(completed_goals.scalars().all())
    
    # Get vector index stats
    index_stats = vector_service.get_index_stats()
    
    return {
        "city_name": city_name,
        "goals": {
            "active": active_count,
            "completed": completed_count,
            "total": active_count + completed_count
        },
        "vector_index": {
            "total_vectors": index_stats['total_vectors'],
            "goals_in_index": index_stats['goals_count'],
            "policies_in_index": index_stats['policies_count']
        },
        "training_status": "trained" if active_count > 0 else "not_trained"
    }

@router.post("/goals/{city_name}/synthesize")
async def synthesize_goal_aligned_report(
    city_name: str,
    request: RAGQueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate a comprehensive report with goal-aligned recommendations"""
    
    try:
        # Get city goals
        goals_result = await db.execute(
            select(CityGoal).where(and_(CityGoal.city_name == city_name, CityGoal.status == "active"))
        )
        city_goals = goals_result.scalars().all()
        
        if not city_goals:
            raise HTTPException(status_code=404, detail=f"No active goals found for {city_name}")
        
        # Get goal-aligned recommendations
        recommendations = vector_service.get_goal_aligned_recommendations(
            city_name=city_name,
            problem_description=request.problem_description,
            current_data=request.current_data,
            max_recommendations=request.max_recommendations
        )
        
        # Create comprehensive synthesis using LLM
        synthesis_prompt = f"""
        You are an urban AI assistant. The city's goals are:
        {[f"- {goal.goal_title}: {goal.goal_description}" for goal in city_goals]}
        
        Given the following current problems and data:
        Problem: {request.problem_description}
        Data: {json.dumps(request.current_data, default=str)}
        
        And these goal-aligned recommendations:
        {json.dumps([r['city_goal']['metadata'] for r in recommendations], default=str)}
        
        Generate a comprehensive report that:
        1. Prioritizes solutions aligned with the city's stated goals
        2. Provides specific implementation steps
        3. Estimates realistic impact and timeline
        4. Considers resource requirements and partnerships needed
        
        Focus on sustainable and community-focused solutions that directly support the city's goals.
        """
        
        # Use Claude to synthesize the final report
        synthesis_result = await claude_service.synthesize_problems(
            {"synthesis_context": synthesis_prompt}, 
            f"Goal-aligned synthesis for {city_name}"
        )
        
        return {
            "city_name": city_name,
            "city_goals": [CityGoalResponse(
                id=goal.id,
                city_name=goal.city_name,
                goal_title=goal.goal_title,
                goal_description=goal.goal_description,
                target_metric=goal.target_metric,
                target_value=goal.target_value,
                target_unit=goal.target_unit,
                priority_level=goal.priority_level,
                deadline=goal.deadline,
                status=goal.status,
                created_at=goal.created_at,
                updated_at=goal.updated_at
            ) for goal in city_goals],
            "recommendations": recommendations,
            "synthesis": synthesis_result,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error synthesizing report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to synthesize report: {str(e)}")

