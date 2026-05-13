import logging
import os
import sys
import time
from typing import Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from components.chunking import SemanticProcessor
from components.generation import Generator
from components.guardrails import Guardrails
from components.indexing import IndexingEngine
from components.ingestion import DocumentIngestor
from components.post_processing import ContextOptimizer
from components.retrieval import HybridRetriever
from config.llm_provider import LLMProvider
from evals.evaluator import RAGEvaluator
from evals.ragas_eval import RagasEvaluator
from utils import setup_logging, timer, suppress_noisy_loggers, clear_timings, get_timings

logger = logging.getLogger(__name__)
console = Console()


class RAGPipeline:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.llm = LLMProvider.get_llm()
        self.ingestor = DocumentIngestor(file_path)
        self.processor = SemanticProcessor()
        self.indexer = IndexingEngine()
        self.raw_docs = None
        self.chunks = None
        self.vector_store = None
        self.bm25 = None
        self.retriever_engine = None
        self.optimizer = None
        self.generator = None

    @timer(name="Load PDF & clean")
    def load(self) -> list:
        self.raw_docs = self.ingestor.load_and_clean()
        return self.raw_docs

    @timer(name="Chunking")
    def chunk(self) -> list:
        self.chunks = self.processor.split(self.raw_docs)
        return self.chunks

    @timer(name="Indexing")
    def index(self) -> None:
        self.vector_store = self.indexer.build_vector_store(self.chunks)
        self.bm25 = self.indexer.build_bm25_index(self.chunks)
        self.retriever_engine = HybridRetriever(self.vector_store, self.bm25, llm=self.llm)
        self.optimizer = ContextOptimizer(self.retriever_engine.ensemble)
        self.generator = Generator(self.llm)

    def ask(self, query: str) -> str:
        return self.run_query(query).get("answer", "")

    @timer(name="Query Execution")
    def run_query(self, query: str, ground_truth: Optional[str] = None) -> dict:
        try:
            retrieved_docs = self.optimizer.get_optimized_docs(query)
            raw_answer = self.generator.generate(query, retrieved_docs)
            is_valid, final_answer = Guardrails.validate_output(raw_answer)
        except Exception as e:
            logger.error("Query processing failed: %s", e)
            return {
                "question": query,
                "answer": f"Error: {e}",
                "contexts": [],
                "ground_truth": ground_truth,
            }

        return {
            "question": query,
            "answer": final_answer if is_valid else "Blocked by Guardrails",
            "contexts": [doc.page_content for doc in retrieved_docs],
            "ground_truth": ground_truth,
        }


if __name__ == "__main__":
    setup_logging(console=console)
    suppress_noisy_loggers()
    clear_timings()

    eval_queries = [
        # {
        #     "query": "What is the greatest obstacle to pursuit of mastery?",
        #     "ground_truth": "The greatest obstacle to our pursuit of mastery comes from the emotional drain we experience in dealing with the resistance and manipulations of the people around us.",
        # },
        # {
        #     "query": "What is Original Mind?",
        #     "ground_truth": "It is the mind looked at the world more directly—not through words and received ideas.",
        # },
        {
            "query": "What is the apprenticeship phase?",
            "ground_truth": "The apprenticeship phase is a period of intense learning and practice under a mentor or master.",
        },
        {
            "query": "How does Robert Greene define mastery?",
            "ground_truth": "Mastery is the ability to practice a skill or body of knowledge at a high level through deep practice and creative.",
        },
    ]

    overall_start = time.perf_counter()
    manual_timings: dict[str, float] = {}

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )

    try:
        with progress:
            main_task = progress.add_task("[cyan]RAG Pipeline", total=100)

            # ── Stage 1: Load PDF ──
            console.print()
            console.print("[bold cyan]▸ Load PDF[/]")
            progress.update(main_task, description="[cyan]Loading PDF & cleaning...")
            t0 = time.perf_counter()
            try:
                pipeline = RAGPipeline("Mastery_by_Robert_Greene.pdf")
                pipeline.load()
            except Exception:
                console.print("[red]Failed to load document[/]")
                raise
            manual_timings["Load PDF"] = time.perf_counter() - t0
            progress.update(main_task, advance=10)

            # PDF stats
            file_size_mb = os.path.getsize("Mastery_by_Robert_Greene.pdf") / (1024 * 1024)
            total_words = sum(len(d.page_content.split()) for d in pipeline.raw_docs)
            stats_table = Table(box=box.SIMPLE, show_header=False)
            stats_table.add_column(style="cyan", no_wrap=True)
            stats_table.add_column(style="white")
            stats_table.add_row("File", f"Mastery_by_Robert_Greene.pdf [dim]({file_size_mb:.1f} MB)")
            stats_table.add_row("Words", f"{total_words:,}")
            stats_table.add_row("Pages", f"{len(pipeline.raw_docs)}")
            console.print(Panel(stats_table, title="[bold]Document Stats", border_style="dim"))

            # ── Stage 2: Chunking ──
            console.print("[bold cyan]▸ Chunking[/]")
            progress.update(main_task, description="[cyan]Chunking...")
            t0 = time.perf_counter()
            try:
                pipeline.chunk()
            except Exception:
                console.print("[red]Chunking failed[/]")
                raise
            manual_timings["Chunking"] = time.perf_counter() - t0
            progress.update(main_task, advance=10)

            # ── Stage 3: Indexing ──
            console.print("[bold cyan]▸ Indexing[/]")
            progress.update(main_task, description="[cyan]Building vector store & BM25 index...")
            t0 = time.perf_counter()
            try:
                pipeline.index()
            except Exception:
                console.print("[red]Indexing failed[/]")
                raise
            manual_timings["Indexing"] = time.perf_counter() - t0
            progress.update(main_task, advance=20)

            # ── Stage 4: Run evaluation queries ──
            console.print("[bold cyan]▸ Running evaluation queries[/]")
            results = []
            for i, item in enumerate(eval_queries):
                q_short = item["query"][:60]
                console.print(f"  [cyan]Query {i + 1}/{len(eval_queries)}:[/] {item['query']}")
                progress.update(
                    main_task,
                    description=f"[cyan]Evaluating query {i + 1}/{len(eval_queries)}: {q_short}...",
                )
                t0 = time.perf_counter()
                try:
                    res = pipeline.run_query(item["query"], item["ground_truth"])
                    results.append(res)
                except Exception as e:
                    logger.error("Eval query failed (%s): %s", item["query"], e)
                manual_timings[f"Query {i + 1}/{len(eval_queries)}"] = time.perf_counter() - t0
                progress.update(main_task, advance=10)

            # ── Stage 5: Keyword evaluation ──
            console.print("[bold cyan]▸ Keyword evaluation[/]")
            progress.update(
                main_task,
                description="[cyan]Running keyword-based evaluation...",
            )
            t0 = time.perf_counter()
            keyword_results: list[tuple[str, str, bool]] = []
            try:
                kw_eval = RAGEvaluator(pipeline)
                for case in kw_eval.test_cases:
                    try:
                        answer = pipeline.ask(case["query"])
                        passed = case["expected_keyword"].lower() in answer.lower()
                        keyword_results.append((case["query"], case["expected_keyword"], passed))
                    except Exception as e:
                        logger.error("Keyword eval failed for '%s': %s", case["query"], e)
            except Exception as e:
                logger.error("Keyword evaluator init failed: %s", e)
            manual_timings["Keyword Evaluation"] = time.perf_counter() - t0
            progress.update(main_task, advance=10)

            # ── Stage 6: RAGAS evaluation ──
            console.print("[bold cyan]▸ RAGAS evaluation[/]")
            progress.update(main_task, description="[cyan]Running RAGAS evaluation...")
            t0 = time.perf_counter()
            ragas_eval = RagasEvaluator()
            summary_df = None
            try:
                summary_df = ragas_eval.run_evaluation(results)
            except Exception as e:
                logger.error("RAGAS evaluation failed: %s", e)
            manual_timings["RAGAS Evaluation"] = time.perf_counter() - t0
            progress.update(main_task, advance=10)

            progress.update(main_task, description="[green]Done!", completed=100)

    except KeyboardInterrupt:
        console.print()
        console.print("[bold yellow]Interrupted by user[/]")
        sys.exit(0)

    # ────────────────────────────────────────────────────────────
    #  Timing summary table
    # ────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - overall_start

    timer_table = Table(
        title="RAG Pipeline - Execution Summary",
        box=box.ROUNDED,
        title_style="bold cyan",
    )
    timer_table.add_column("Stage", style="cyan", no_wrap=True)
    timer_table.add_column("Time (s)", justify="right", style="green")
    timer_table.add_column("% of Total", justify="right", style="yellow")

    for stage, elapsed in manual_timings.items():
        pct = (elapsed / total_time) * 100
        timer_table.add_row(stage, f"{elapsed:.2f}", f"{pct:.1f}%")

    all_timings = get_timings()
    sub_stages = [
        # Under Load PDF
        "Initialize LLM",
        "Load PDF & clean",
        # Under Chunking
        "Recursive split",
        "Semantic chunking",
        "Enrich metadata",
        # Under Indexing
        "Build vector store",
        "Build BM25 index",
        # Under each query
        "Hybrid retrieval",
        "Rerank & compress",
        "LLM generation",
        "Output validation",
    ]
    for label in sub_stages:
        if label in all_timings:
            timer_table.add_row(f"  \u2514\u2500 {label}", f"{all_timings[label]:.2f}", "")

    timer_table.add_row("\u2500" * 30, "\u2500" * 10, "\u2500" * 10, style="dim")
    timer_table.add_row("[bold]TOTAL", f"[bold]{total_time:.2f}", "[bold]100.0%")

    console.print()
    console.print(timer_table)
    console.print()

    # ────────────────────────────────────────────────────────────
    #  RAGAS scores table
    # ────────────────────────────────────────────────────────────
    if summary_df is not None and not summary_df.empty:
        score_table = Table(
            title="RAGAS Evaluation Scores",
            box=box.ROUNDED,
            title_style="bold cyan",
        )
        score_table.add_column("Metric", style="cyan")
        score_table.add_column("Mean", justify="right", style="green")
        score_table.add_column("Per Query", style="white")

        for col in summary_df.select_dtypes(include="number").columns:
            values = summary_df[col]
            mean = values.mean()
            per_query = ", ".join(f"{v:.4f}" for v in values)
            score_table.add_row(col, f"{mean:.4f}", per_query)

        console.print(score_table)
        console.print()

    # ────────────────────────────────────────────────────────────
    #  Keyword evaluation results
    # ────────────────────────────────────────────────────────────
    if keyword_results:
        kw_table = Table(
            title="Keyword-Based Evaluation",
            box=box.ROUNDED,
            title_style="bold cyan",
        )
        kw_table.add_column("Query", style="cyan", no_wrap=False)
        kw_table.add_column("Expected Keyword", style="yellow")
        kw_table.add_column("Result", justify="center")

        for q, kw, passed in keyword_results:
            status = "[green]PASS[/]" if passed else "[red]FAIL[/]"
            kw_table.add_row(q[:80] + ("..." if len(q) > 80 else ""), kw, status)

        console.print(kw_table)
        console.print()
