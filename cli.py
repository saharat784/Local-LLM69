import sys
import os

# Fix Windows console UTF-8 output issue
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import ollama
from rich.console import Console

from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.live import Live
from rag_engine import RAGEngine

console = Console()

def get_available_models():
    try:
        models_response = ollama.list()
        models = [m.model for m in models_response.models]
        return models
    except Exception as e:
        console.print(f"[bold red]Error connecting to Ollama service:[/bold red] {e}")
        console.print("[yellow]Make sure Ollama is running (`ollama serve`).[/yellow]")
        sys.exit(1)

def main():
    console.print(Panel.fit("[bold cyan]🤖 Ollama LLM + RAG CLI Chat[/bold cyan]\nPowered by Ollama, bge-m3 & ChromaDB", border_style="cyan"))

    models = get_available_models()
    if not models:
        console.print("[bold red]No local models found![/bold red] Please run `ollama pull <model_name>` first.")
        sys.exit(1)

    console.print("[bold green]Available Models:[/bold green]")
    for idx, model in enumerate(models, 1):
        console.print(f"  {idx}. [bold yellow]{model}[/bold yellow]")

    # Select model
    if "qwen2.5-coder:7b" in models:
        selected_model = "qwen2.5-coder:7b"
        console.print(f"\n[dim]Auto-selected model:[/dim] [bold yellow]{selected_model}[/bold yellow]")
    elif len(models) == 1:
        selected_model = models[0]
        console.print(f"\n[dim]Selected default model:[/dim] [bold yellow]{selected_model}[/bold yellow]")
    else:
        choice = Prompt.ask("\nSelect model number", choices=[str(i) for i in range(1, len(models) + 1)], default="1")
        selected_model = models[int(choice) - 1]

    # Initialize RAG
    rag_engine = RAGEngine()
    enable_rag = Confirm.ask("\nEnable Knowledge Base RAG Search?", default=True)

    if enable_rag:
        with console.status("[bold green]Indexing knowledge base documents..."):
            stats = rag_engine.index_knowledge_base()
            console.print(f"[dim]Indexed {stats['files_indexed']} files ({stats['total_chunks']} chunks)[/dim]")

    console.print(f"\n[bold green]Chatting with [yellow]{selected_model}[/yellow] (RAG: {'[green]ON[/green]' if enable_rag else '[red]OFF[/red]'}). Type [bold red]'exit'[/bold red] to quit.[/bold green]\n")

    messages = []
    base_system_prompt = "You are a helpful, accurate, and concise AI assistant."

    while True:
        try:
            user_input = console.input("[bold cyan]You > [/bold cyan]").strip()
            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("[bold yellow]Goodbye! 👋[/bold yellow]")
                break

            messages.append({'role': 'user', 'content': user_input})

            effective_system_prompt = base_system_prompt
            retrieved_chunks = []

            if enable_rag:
                with console.status("[dim]Searching knowledge base...[/dim]"):
                    retrieved_chunks = rag_engine.retrieve(user_input, top_k=4)

                if retrieved_chunks:
                    console.print(f"[dim]📚 Found {len(retrieved_chunks)} relevant source context(s):[/dim]")
                    for src in retrieved_chunks:
                        console.print(f"  • [yellow]{src['source']}[/yellow] (Similarity: {src['score']})")

                effective_system_prompt = rag_engine.build_rag_system_prompt(base_system_prompt, retrieved_chunks)

            # Build full message payload including system prompt
            api_messages = [{'role': 'system', 'content': effective_system_prompt}] + messages

            console.print(f"\n[bold magenta]{selected_model}[/bold magenta] > ", end="")
            
            full_response = ""
            with Live("", console=console, refresh_per_second=10) as live:
                stream = ollama.chat(
                    model=selected_model,
                    messages=api_messages,
                    stream=True,
                )
                for chunk in stream:
                    content = chunk.get('message', {}).get('content', '')
                    full_response += content
                    live.update(Markdown(full_response))

            messages.append({'role': 'assistant', 'content': full_response})
            console.print()  # newline after response

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Chat session ended. Goodbye! 👋[/bold yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error during response generation:[/bold red] {e}")

if __name__ == "__main__":
    main()
