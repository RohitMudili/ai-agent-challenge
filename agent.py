#!/usr/bin/env python3
"""
Agent-as-Coder: Autonomous PDF Parser Generator
Builds custom bank statement parsers using LLM-powered agent loop.
"""

import os
import sys
import argparse
from typing import TypedDict, Annotated, Literal
from pathlib import Path
import pandas as pd
import pdfplumber
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """State maintained throughout the agent workflow."""
    target_bank: str
    pdf_path: str
    csv_path: str
    parser_path: str
    pdf_sample: str
    csv_sample: str
    generated_code: str
    test_output: str
    error_message: str
    attempt: int
    max_attempts: int
    status: Literal["planning", "coding", "testing", "fixing", "success", "failed"]


class PDFParserAgent:
    """Autonomous agent that generates bank statement parsers."""

    def __init__(self, llm_provider: str = "openai"):
        """Initialize the agent with LLM provider."""
        if llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        elif llm_provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for the agent."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("generate_code", self._generate_code_node)
        workflow.add_node("test_code", self._test_code_node)
        workflow.add_node("fix_code", self._fix_code_node)
        workflow.add_node("finalize", self._finalize_node)

        # Define edges
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "generate_code")
        workflow.add_edge("generate_code", "test_code")
        workflow.add_conditional_edges(
            "test_code",
            self._should_fix_or_finish,
            {
                "fix": "fix_code",
                "success": "finalize",
                "failed": "finalize"
            }
        )
        workflow.add_edge("fix_code", "test_code")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _plan_node(self, state: AgentState) -> AgentState:
        """Plan the parser implementation by analyzing PDF structure."""
        print(f"\n📋 Planning parser for {state['target_bank'].upper()}...")

        # Extract sample from PDF
        pdf_sample = self._extract_pdf_sample(state['pdf_path'])
        csv_sample = self._read_csv_sample(state['csv_path'])

        state['pdf_sample'] = pdf_sample
        state['csv_sample'] = csv_sample
        state['status'] = "planning"
        state['attempt'] = 0

        print(f"✓ Analyzed PDF structure and CSV schema")
        return state

    def _generate_code_node(self, state: AgentState) -> AgentState:
        """Generate the parser code using LLM."""
        print(f"\n💻 Generating parser code (Attempt {state['attempt'] + 1}/{state['max_attempts']})...")

        prompt = self._create_generation_prompt(state)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        # Extract code from response
        code = self._extract_code_from_response(response.content)
        state['generated_code'] = code
        state['status'] = "coding"

        # Save code to file
        with open(state['parser_path'], 'w') as f:
            f.write(code)

        print(f"✓ Generated parser saved to {state['parser_path']}")
        return state

    def _test_code_node(self, state: AgentState) -> AgentState:
        """Test the generated parser."""
        print(f"\n🧪 Testing generated parser...")

        try:
            # Import the generated parser dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location("test_parser", state['parser_path'])
            parser_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(parser_module)

            # Run the parser
            result_df = parser_module.parse(state['pdf_path'])

            # Load expected CSV
            expected_df = pd.read_csv(state['csv_path'])

            # Compare results
            if self._validate_dataframes(result_df, expected_df):
                state['status'] = "success"
                state['test_output'] = "✓ All tests passed! Parser output matches expected CSV."
                state['error_message'] = ""
                print("✓ All tests passed!")
            else:
                diff_msg = self._get_dataframe_diff(result_df, expected_df)
                state['status'] = "testing"
                state['error_message'] = f"DataFrame mismatch:\n{diff_msg}"
                print(f"✗ Test failed: DataFrame mismatch")

        except Exception as e:
            state['status'] = "testing"
            state['error_message'] = f"Error during testing: {str(e)}\n{type(e).__name__}"
            print(f"✗ Test failed: {str(e)}")

        state['attempt'] += 1
        return state

    def _fix_code_node(self, state: AgentState) -> AgentState:
        """Fix the parser code based on test errors."""
        print(f"\n🔧 Fixing parser code...")

        prompt = self._create_fix_prompt(state)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        # Extract fixed code
        code = self._extract_code_from_response(response.content)
        state['generated_code'] = code
        state['status'] = "fixing"

        # Save fixed code
        with open(state['parser_path'], 'w') as f:
            f.write(code)

        print(f"✓ Updated parser with fixes")
        return state

    def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize the agent workflow."""
        if state['status'] == "success":
            print(f"\n✅ SUCCESS! Parser generated at {state['parser_path']}")
        else:
            print(f"\n❌ FAILED after {state['max_attempts']} attempts")
            print(f"Last error: {state['error_message']}")
        return state

    def _should_fix_or_finish(self, state: AgentState) -> str:
        """Decide whether to fix code, finish successfully, or fail."""
        if state['status'] == "success":
            return "success"
        elif state['attempt'] >= state['max_attempts']:
            return "failed"
        else:
            return "fix"

    def _extract_pdf_sample(self, pdf_path: str, max_rows: int = 10) -> str:
        """Extract sample data from PDF."""
        with pdfplumber.open(pdf_path) as pdf:
            first_page = pdf.pages[0]
            tables = first_page.extract_tables()
            if tables:
                sample = "\n".join([" | ".join(map(str, row)) for row in tables[0][:max_rows]])
                return f"PDF Table Sample (first {max_rows} rows):\n{sample}"
            else:
                text = first_page.extract_text()
                return f"PDF Text Sample:\n{text[:1000]}"

    def _read_csv_sample(self, csv_path: str, max_rows: int = 5) -> str:
        """Read sample from CSV."""
        df = pd.read_csv(csv_path)
        return f"Expected CSV Schema:\nColumns: {list(df.columns)}\nDtypes: {df.dtypes.to_dict()}\n\nSample rows:\n{df.head(max_rows).to_string()}"

    def _create_generation_prompt(self, state: AgentState) -> str:
        """Create prompt for initial code generation."""
        return f"""You are an expert Python developer specializing in PDF parsing. Generate a complete, production-ready parser for {state['target_bank'].upper()} bank statements.

**PDF Structure:**
{state['pdf_sample']}

**Expected Output:**
{state['csv_sample']}

**Requirements:**
1. Create a function `parse(pdf_path: str) -> pd.DataFrame` that:
   - Takes a PDF path as input
   - Returns a pandas DataFrame matching the exact CSV schema
   - Handles multiple pages if present
   - Uses pdfplumber for PDF extraction

2. The DataFrame MUST have these exact columns: Date, Description, Debit Amt, Credit Amt, Balance

3. Data types and formatting:
   - Date: string in format DD-MM-YYYY
   - Description: string
   - Debit Amt: float (empty string if no debit)
   - Credit Amt: float (empty string if no credit)
   - Balance: float

4. Important parsing rules:
   - Extract data from table structure in PDF
   - Handle empty cells correctly (use empty string for missing Debit/Credit)
   - Preserve exact numeric values
   - Skip header rows and any footer text
   - Combine data from all pages if multi-page PDF

**Code Quality:**
- Add type hints
- Include docstrings
- Handle edge cases
- Use proper error handling

Generate ONLY the Python code with no explanations. Start with imports and end with the parse() function.
"""

    def _create_fix_prompt(self, state: AgentState) -> str:
        """Create prompt for fixing code."""
        return f"""The parser code has errors. Fix them to make tests pass.

**Current Code:**
```python
{state['generated_code']}
```

**Error/Test Failure:**
{state['error_message']}

**Expected Output Schema:**
{state['csv_sample']}

**Instructions:**
1. Analyze the error carefully
2. Fix the code to handle the issue
3. Ensure the output DataFrame exactly matches the expected CSV structure
4. Return the complete fixed code (not just the changes)

Generate ONLY the corrected Python code with no explanations.
"""

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code blocks
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + len("```")
            end = response.find("```", start)
            return response[start:end].strip()
        else:
            # Return as-is if no code blocks
            return response.strip()

    def _validate_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        """Validate if two DataFrames are equal."""
        try:
            # Check shape
            if df1.shape != df2.shape:
                return False

            # Check columns
            if list(df1.columns) != list(df2.columns):
                return False

            # Convert to same types for comparison
            df1_compare = df1.copy()
            df2_compare = df2.copy()

            # Handle empty strings vs NaN
            for col in df1_compare.columns:
                df1_compare[col] = df1_compare[col].replace('', pd.NA)
                df2_compare[col] = df2_compare[col].replace('', pd.NA)

            # Compare values with tolerance for floats
            return df1_compare.equals(df2_compare)

        except Exception:
            return False

    def _get_dataframe_diff(self, df1: pd.DataFrame, df2: pd.DataFrame) -> str:
        """Get human-readable difference between DataFrames."""
        issues = []

        if df1.shape != df2.shape:
            issues.append(f"Shape mismatch: got {df1.shape}, expected {df2.shape}")

        if list(df1.columns) != list(df2.columns):
            issues.append(f"Column mismatch: got {list(df1.columns)}, expected {list(df2.columns)}")

        # Sample comparison
        issues.append(f"\nFirst 3 rows of generated output:\n{df1.head(3).to_string()}")
        issues.append(f"\nFirst 3 rows of expected output:\n{df2.head(3).to_string()}")

        return "\n".join(issues)

    def run(self, target_bank: str, max_attempts: int = 3) -> bool:
        """Run the agent to generate a parser."""
        # Set up paths
        data_dir = Path("data") / target_bank
        pdf_path = list(data_dir.glob("*.pdf"))[0]
        csv_path = data_dir / "result.csv"
        parser_path = Path("custom_parsers") / f"{target_bank}_parser.py"

        # Initialize state
        initial_state: AgentState = {
            "target_bank": target_bank,
            "pdf_path": str(pdf_path),
            "csv_path": str(csv_path),
            "parser_path": str(parser_path),
            "pdf_sample": "",
            "csv_sample": "",
            "generated_code": "",
            "test_output": "",
            "error_message": "",
            "attempt": 0,
            "max_attempts": max_attempts,
            "status": "planning"
        }

        print(f"\n{'='*60}")
        print(f"🤖 Agent-as-Coder: PDF Parser Generator")
        print(f"{'='*60}")
        print(f"Target Bank: {target_bank.upper()}")
        print(f"PDF: {pdf_path}")
        print(f"Expected CSV: {csv_path}")
        print(f"Output: {parser_path}")
        print(f"Max Attempts: {max_attempts}")
        print(f"{'='*60}")

        # Run the workflow
        final_state = self.graph.invoke(initial_state)

        return final_state["status"] == "success"


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Agent-as-Coder: Generate PDF parsers autonomously")
    parser.add_argument("--target", required=True, help="Target bank name (e.g., icici, sbi)")
    parser.add_argument("--llm", default="openai", choices=["openai", "gemini"], help="LLM provider to use")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum fix attempts")

    args = parser.parse_args()

    try:
        agent = PDFParserAgent(llm_provider=args.llm)
        success = agent.run(args.target, max_attempts=args.max_attempts)
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ Agent failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
