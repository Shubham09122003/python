from robyn import Robyn, jsonify, Request, Response
# from robyn.authentication import BearerAuthentication
import asyncio
import motor.motor_asyncio
import redis.asyncio as redis
import json
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import openai
import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Initialize Robyn app
app = Robyn(__file__)

# Database connections
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = mongo_client.git_analytics
redis_client = redis.Redis(host=os.getenv("REDIS_HOST"), 
                          port=os.getenv("REDIS_PORT"), 
                          password=os.getenv("REDIS_PASSWORD"),
                          decode_responses=True)

# OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Authentication middleware
# bearer_auth = BearerAuthentication(
#     lambda token: token == os.getenv("API_TOKEN"),
#     excluded_paths=["/health", "/webhook"]
# )
# app.use(bearer_auth)

# Models
class UserProfile(BaseModel):
    username: str
    bio: Optional[str]
    profile_pic: Optional[str]
    repositories: int
    public_repos: int
    private_repos: int
    followers: int
    following: int
    top_languages: List[str]

class CommitAnalysis(BaseModel):
    total_commits: int
    lines_added: int
    lines_deleted: int
    commit_activity: Dict[str, int]
    most_frequent_commit_messages: List[str]
    files_modified: List[str]

class RepoHealth(BaseModel):
    repo_name: str
    open_prs: int
    closed_prs: int
    avg_issue_resolution_time: str
    fork_count: int
    star_count: int
    code_reviews: Dict[str, int]

class Collaboration(BaseModel):
    username: str
    pull_requests: Dict[str, int]
    issues_resolved: int
    bug_fixes: int

class AIInsights(BaseModel):
    username: str
    code_churn_rate: str
    repeated_code_patterns: List[str]
    improvement_suggestions: List[str]
    developer_ranking: str

class WebhookEvent(BaseModel):
    repo: str
    branch: str
    commit_id: str
    author: str
    message: str
    files_modified: List[str]
    timestamp: str

class LeaderboardEntry(BaseModel):
    top_contributors: List[str]
    fastest_issue_resolvers: List[str]
    most_active_developers: List[str]

# Helper functions
async def fetch_github_data(username):
    """Fetch GitHub data for a user using the GitHub API"""
    async with aiohttp.ClientSession() as session:
        # Base user info
        async with session.get(f"https://api.github.com/users/{username}", 
                            headers={"Accept": "application/vnd.github.v3+json"}) as resp:
            if resp.status != 200:
                return None
            user_data = await resp.json()
        
        # Repos
        async with session.get(f"https://api.github.com/users/{username}/repos", 
                            headers={"Accept": "application/vnd.github.v3+json"}) as resp:
            if resp.status != 200:
                return None
            repos = await resp.json()
        
        # Extract languages
        languages = {}
        for repo in repos[:10]:  # Limit to 10 repos to avoid rate limits
            if repo["language"] and repo["language"] not in languages:
                languages[repo["language"]] = 1
            elif repo["language"]:
                languages[repo["language"]] += 1
        
        top_languages = sorted(languages.keys(), key=lambda l: languages[l], reverse=True)[:5]
        
        return {
            "username": username,
            "bio": user_data.get("bio", ""),
            "profile_pic": user_data.get("avatar_url", ""),
            "repositories": user_data.get("public_repos", 0),
            "public_repos": user_data.get("public_repos", 0),
            "private_repos": 0,  # Private repo count needs auth
            "followers": user_data.get("followers", 0),
            "following": user_data.get("following", 0),
            "top_languages": top_languages
        }

async def analyze_commits(repo_name, branch="main"):
    """Analyze commits for a repository"""
    # In a real app, this would use the GitHub API to fetch actual commit data
    # This is a simplified mock
    
    # Simulate getting data from cache or database
    cached = await redis_client.get(f"commits:{repo_name}:{branch}")
    if cached:
        return json.loads(cached)
    
    # Mock data
    analysis = {
        "total_commits": 540,
        "lines_added": 12540,
        "lines_deleted": 4300,
        "commit_activity": {
            "morning_commits": 50,
            "afternoon_commits": 250,
            "night_commits": 240
        },
        "most_frequent_commit_messages": ["fix bug", "update readme", "refactor"],
        "files_modified": ["app.py", "models.py", "utils.js"]
    }
    
    # Cache result
    await redis_client.set(f"commits:{repo_name}:{branch}", json.dumps(analysis), ex=3600)
    
    return analysis

async def get_repo_health(repo_name):
    """Get repository health metrics"""
    # Simplified mock
    health = {
        "repo_name": repo_name,
        "open_prs": 12,
        "closed_prs": 85,
        "avg_issue_resolution_time": "2.5 days",
        "fork_count": 120,
        "star_count": 900, 
        "code_reviews": {
            "completed_reviews": 60,
            "pending_reviews": 5
        }
    }
    
    return health

async def get_collaboration_metrics(username):
    """Get collaboration metrics for a user"""
    # Mock data
    metrics = {
        "username": username,
        "pull_requests": {
            "approved": 45,
            "merged": 38
        },
        "issues_resolved": 22,
        "bug_fixes": 15
    }
    
    return metrics

async def generate_ai_insights(username):
    """Generate AI-powered insights for a user"""
    # In a real app, this would use OpenAI or a custom ML model
    
    # Mock data
    insights = {
        "username": username,
        "code_churn_rate": "8.3%",
        "repeated_code_patterns": ["function duplicate in utils.py and helpers.py"],
        "improvement_suggestions": ["Optimize API calls to reduce latency"],
        "developer_ranking": "Top 5% globally"
    }
    
    # In a real implementation, this would use OpenAI like:
    # completion = await openai.ChatCompletion.acreate(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": "You are a code review assistant."},
    #         {"role": "user", "content": f"Analyze the following code changes by user {username}..."}
    #     ]
    # )
    # insights["improvement_suggestions"] = [completion.choices[0].message.content]
    
    return insights

async def get_leaderboard():
    """Get developer leaderboard"""
    # Mock data
    leaderboard = {
        "top_contributors": ["dev123", "codeMaster", "aiGuru"],
        "fastest_issue_resolvers": ["fixitQuick", "bugHunter"],
        "most_active_developers": ["nightCoder", "earlyBird"]
    }
    
    return leaderboard

async def process_webhook_event(event):
    """Process a webhook event"""
    # Store in MongoDB
    await db.webhook_events.insert_one(event)
    
    # Publish to Redis for real-time updates
    await redis_client.publish("git_events", json.dumps(event))
    
    # Update cache for affected repositories
    await redis_client.delete(f"commits:{event['repo']}:{event['branch']}")
    
    return True

# API Routes

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

@app.get("/api/user/:username")
async def get_user_profile(request: Request, username: str):
    """Get user profile data"""
    # Check cache first
    print("{request.path_params.get('username')}")
    cached = await redis_client.get(f"user:{request.path_params.get('username')}")
    if cached:
        return jsonify(json.loads(cached))
    
    # Fetch from GitHub
    profile = await fetch_github_data(username)
    if not profile:
        return Response(status_code=404, body=jsonify({"error": "User not found"}))
    
    # Cache result
    await redis_client.set(f"user:{username}", json.dumps(profile), ex=3600)
    
    return jsonify(profile)

@app.get("/api/commits/{repo_name}/{branch}")
async def get_commits(request: Request, repo_name: str, branch: str):
    """Get commit analysis for a repository"""
    analysis = await analyze_commits(repo_name, branch)
    return jsonify(analysis)

@app.get("/api/repo/{repo_name}/health")
async def get_repo_health_metrics(request: Request, repo_name: str):
    """Get repository health metrics"""
    health = await get_repo_health(repo_name)
    return jsonify(health)

@app.get("/api/user/{username}/collaboration")
async def get_user_collaboration(request: Request, username: str):
    """Get collaboration metrics for a user"""
    metrics = await get_collaboration_metrics(username)
    return jsonify(metrics)

@app.get("/api/user/{username}/insights")
async def get_user_insights(request: Request, username: str):
    """Get AI-powered insights for a user"""
    insights = await generate_ai_insights(username)
    return jsonify(insights)

@app.get("/api/leaderboard")
async def get_developer_leaderboard(request: Request):
    """Get developer leaderboard"""
    leaderboard = await get_leaderboard()
    return jsonify(leaderboard)

@app.post("/api/webhook/commits")
async def webhook_handler(request: Request):
    """Handle webhook events from Git providers"""
    try:
        body = await request.json()
        event = WebhookEvent(**body)
        success = await process_webhook_event(event.dict())
        if success:
            return jsonify({"status": "received", "event_id": str(event.commit_id)})
        else:
            return Response(status_code=500, body=jsonify({"error": "Failed to process webhook"}))
    except Exception as e:
        return Response(status_code=400, body=jsonify({"error": str(e)}))

# Start server
if __name__ == "__main__":
    app.start()