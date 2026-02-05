#!/usr/bin/env python3
"""
Main Application Entry Point
"""

import argparse
from pathlib import Path
from agent_core import DocumentQAAgent
from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Document Q&A Agent with Skill Integration")
    parser.add_argument("--document", "-d", help="Document file to process")
    parser.add_argument("--query", "-q", help="Question to ask about the document")
    parser.add_argument("--questions-file", help="File with questions (one per line)")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Start interactive session")
    parser.add_argument("--list-skills", action="store_true", 
                       help="List available skills")
    parser.add_argument("--skill-dir", default="skills", 
                       help="Directory containing skills")
    
    args = parser.parse_args()
    
    # Initialize agent
    console.print("[bold blue]🤖 Initializing Document Q&A Agent...[/bold blue]")
    agent = DocumentQAAgent(skills_dir=args.skill_dir)
    
    # List skills if requested
    if args.list_skills:
        console.print("\n[bold]📚 Available Skills:[/bold]")
        for skill_name, skill_data in agent.skill_manager.skills.items():
            metadata = skill_data['metadata']
            console.print(f"  • [cyan]{metadata.name}[/cyan]: {metadata.description}")
        return
    
    # Interactive mode
    if args.interactive:
        agent.interactive_session()
        return
    
    # Single document + question
    if args.document and args.query:
        console.print(f"[bold]Processing: {Path(args.document).name}[/bold]")
        
        # Process document
        doc_result = agent.process_document(args.document)
        if 'error' in doc_result:
            console.print(f"[red]Error: {doc_result['error']}[/red]")
            return
        
        # Ask question
        context = {
            'user_query': args.query,
            'document_text': agent.processed_documents[doc_result['document_id']]['content']
        }
        
        response = agent.process_with_skills(args.query, context)
        
        console.print(Panel(
            response,
            title=f"Answer to: {args.query}",
            border_style="green"
        ))
        
        return
    
    # Batch questions from file
    if args.document and args.questions_file:
        console.print(f"[bold]Batch processing: {Path(args.document).name}[/bold]")
        
        # Process document
        doc_result = agent.process_document(args.document)
        if 'error' in doc_result:
            console.print(f"[red]Error: {doc_result['error']}[/red]")
            return
        
        # Read questions
        with open(args.questions_file, 'r') as f:
            questions = [q.strip() for q in f if q.strip()]
        
        console.print(f"[green]Loaded {len(questions)} questions[/green]")
        
        # Process each question
        for i, question in enumerate(questions, 1):
            console.print(f"\n[cyan]Question {i}/{len(questions)}:[/cyan] {question}")
            
            context = {
                'user_query': question,
                'document_text': agent.processed_documents[doc_result['document_id']]['content']
            }
            
            response = agent.process_with_skills(question, context)
            
            console.print(Panel(
                response[:500] + "..." if len(response) > 500 else response,
                title=f"Answer {i}",
                border_style="blue"
            ))
        
        return
    
    # No valid arguments, show help
    console.print(Panel(
        "[bold]Document Q&A Agent[/bold]\n\n"
        "Usage modes:\n"
        "1. Interactive: python main_app.py --interactive\n"
        "2. Single question: python main_app.py --document file.pdf --query \"Your question\"\n"
        "3. Batch questions: python main_app.py --document file.pdf --questions-file questions.txt\n"
        "4. List skills: python main_app.py --list-skills",
        title="Help",
        border_style="yellow"
    ))

if __name__ == "__main__":
    main()