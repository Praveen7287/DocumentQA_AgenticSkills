#!/usr/bin/env python3
"""
Agent Core with Skill Integration
Main agent that uses skills dynamically
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import ollama
from skill_manager import SkillManager
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from config import *

console = Console()

class DocumentQAAgent:
    """Main agent that orchestrates document Q&A using skills"""
    
    def __init__(self, skills_dir: str = "skills"):
        self.skill_manager = SkillManager(skills_dir)
        self.ollama_model = OLLAMA_MODEL
        self.processed_documents = {}
        
        console.print(Panel.fit(
            "[bold blue]🤖 Document Q&A Agent Initialized[/bold blue]",
            border_style="blue"
        ))
        console.print(f"📚 Skills loaded: {len(self.skill_manager.skills)}")
    
    def get_system_prompt(self) -> str:
        """Generate the system prompt with available skills"""
        skills_xml = self.skill_manager.get_available_skills_xml()
        
        return f"""You are an intelligent document analysis assistant with access to specialized skills.

AVAILABLE SKILLS:
{skills_xml}

HOW TO USE SKILLS:
1. Analyze the user's request to determine which skills are needed
2. You can use multiple skills in sequence if needed
3. Each skill provides specific instructions for its task
4. When you decide to use a skill, output: [USE_SKILL:skill_name]
5. After skill execution, continue processing with the results

RESPONSE FORMAT:
- Think step by step
- Indicate which skills you're using
- Provide clear, concise answers
- Cite sources when relevant
- Acknowledge limitations
"""
    
    def process_with_skills(self, user_input: str, context: Dict = None) -> str:
        """
        Process user input using available skills
        
        Args:
            user_input: User's question or request
            context: Additional context (e.g., document content)
            
        Returns:
            Agent's response
        """
        # Prepare the full prompt
        system_prompt = self.get_system_prompt()
        
        user_prompt = f"""USER REQUEST: {user_input}

"""
        if context and 'document_text' in context:
            user_prompt += f"""DOCUMENT CONTEXT:
{context['document_text'][:5000]}...
"""
        
        console.print("[dim]🤔 Agent is thinking...[/dim]")
        
        try:
            # Get initial agent response
            response = ollama.generate(
                model=self.ollama_model,
                system=system_prompt,
                prompt=user_prompt,
                options={
                    'temperature': 0.3,
                    'num_predict': 2000
                }
            )
            
            agent_response = response['response']
            
            # Check if agent wants to use a skill
            if '[USE_SKILL:' in agent_response:
                return self._execute_skill_chain(agent_response, context)
            
            return agent_response
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _execute_skill_chain(self, agent_response: str, context: Dict) -> str:
        """
        Execute skills based on agent's request
        
        Args:
            agent_response: Agent's response containing skill requests
            context: Execution context
            
        Returns:
            Final response after skill execution
        """
        console.print("[yellow]🔧 Agent is using skills...[/yellow]")
        
        # Parse skill requests from agent response
        lines = agent_response.split('\n')
        skill_requests = []
        
        for line in lines:
            if '[USE_SKILL:' in line:
                # Extract skill name from [USE_SKILL:skill_name]
                skill_name = line.split('[USE_SKILL:')[1].split(']')[0].strip()
                skill_requests.append(skill_name)
        
        if not skill_requests:
            return agent_response
        
        # Execute skills in sequence
        workflow_context = context.copy() if context else {}
        results = []
        
        for skill_name in skill_requests:
            console.print(f"  → Executing: {skill_name}")
            
            # Get skill-specific context
            skill_context = {
                'user_query': workflow_context.get('user_query', ''),
                'document_content': workflow_context.get('document_text', ''),
                'previous_results': results,
                'workflow_step': len(results) + 1
            }
            
            # Execute skill
            result = self.skill_manager.activate_skill(skill_name, skill_context)
            results.append({
                'skill': skill_name,
                'result': result
            })
            
            # Update workflow context
            workflow_context[f'result_{skill_name}'] = result
        
        # Generate final answer based on skill results
        final_prompt = f"""SKILL EXECUTION RESULTS:
{json.dumps(results, indent=2)}

ORIGINAL USER REQUEST: {workflow_context.get('user_query', 'Unknown')}

TASK: Synthesize the skill execution results into a clear, comprehensive answer for the user.
Focus on accuracy, completeness, and relevance to the original question.

FINAL ANSWER:"""
        
        try:
            final_response = ollama.generate(
                model=self.ollama_model,
                prompt=final_prompt,
                options={
                    'temperature': 0.2,
                    'num_predict': 1500
                }
            )
            
            return final_response['response']
            
        except Exception as e:
            # Fallback to simple combination
            combined = f"Skill execution completed. Results:\n\n"
            for res in results:
                combined += f"**{res['skill']}**: {res['result'][:500]}...\n\n"
            return combined
    
    def process_document(self, file_path: str) -> Dict:
        """
        Process a document and prepare it for questioning
        
        Args:
            file_path: Path to document
            
        Returns:
            Document processing results
        """
        console.print(f"[bold]📄 Processing document: {Path(file_path).name}[/bold]")
        
        # Read document content (simplified - in production use extract_text.py)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            # Try binary mode for PDFs
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text() + "\n\n"
            except Exception as e:
                return {"error": f"Cannot read document: {str(e)}"}
        
        doc_id = f"doc_{Path(file_path).stem}"
        self.processed_documents[doc_id] = {
            'id': doc_id,
            'path': file_path,
            'content': content,
            'length': len(content)
        }
        
        console.print(f"[green]✓[/green] Document loaded ({len(content):,} chars)")
        
        return {
            'document_id': doc_id,
            'content_preview': content[:500] + "..." if len(content) > 500 else content
        }
    
    def interactive_session(self):
        """Run an interactive Q&A session"""
        console.print(Panel.fit(
            "[bold green]💬 Document Q&A Interactive Session[/bold green]\n"
            "Upload a document and ask questions about it.\n"
            "Type 'quit' to exit, 'new' for new document.",
            border_style="green"
        ))
        
        current_doc = None
        
        while True:
            if not current_doc:
                doc_path = console.input("\n📁 Enter document path: ").strip()
                if doc_path.lower() in ['quit', 'exit']:
                    break
                
                result = self.process_document(doc_path)
                if 'error' in result:
                    console.print(f"[red]{result['error']}[/red]")
                    continue
                
                current_doc = result
                console.print(f"[green]Ready to answer questions about this document![/green]")
            
            # Get user question
            question = console.input("\n❓ Your question (or 'new'/'quit'): ").strip()
            
            if question.lower() in ['quit', 'exit']:
                break
            elif question.lower() == 'new':
                current_doc = None
                continue
            elif not question:
                continue
            
            # Prepare context
            context = {
                'user_query': question,
                'document_text': self.processed_documents[current_doc['document_id']]['content'],
                'document_id': current_doc['document_id']
            }
            
            # Process with agent
            console.print("[cyan]🤖 Processing your question...[/cyan]")
            response = self.process_with_skills(question, context)
            
            # Display response
            console.print(Panel(
                Markdown(response),
                title="Answer",
                border_style="blue",
                width=80
            ))

if __name__ == "__main__":
    # Test the agent
    agent = DocumentQAAgent()
    agent.interactive_session()