"""Test suite for bank statement parsers."""

import pytest
import pandas as pd
from pathlib import Path
import importlib.util


def load_parser(bank_name: str):
    """Dynamically load a parser module."""
    parser_path = Path(f"custom_parsers/{bank_name}_parser.py")
    if not parser_path.exists():
        pytest.skip(f"Parser for {bank_name} not found")

    spec = importlib.util.spec_from_file_location(f"{bank_name}_parser", parser_path)
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)
    return parser_module


class TestICICIParser:
    """Test suite for ICICI bank parser."""

    def test_icici_parser_exists(self):
        """Test that ICICI parser exists."""
        parser_path = Path("custom_parsers/icici_parser.py")
        assert parser_path.exists(), "ICICI parser not found"

    def test_icici_parser_has_parse_function(self):
        """Test that parser has parse() function."""
        parser = load_parser("icici")
        assert hasattr(parser, "parse"), "Parser missing parse() function"

    def test_icici_parser_output_schema(self):
        """Test that parser output has correct schema."""
        parser = load_parser("icici")
        pdf_path = "data/icici/icici sample.pdf"

        result_df = parser.parse(pdf_path)

        # Check columns
        expected_columns = ["Date", "Description", "Debit Amt", "Credit Amt", "Balance"]
        assert list(result_df.columns) == expected_columns, f"Column mismatch. Got: {list(result_df.columns)}"

    def test_icici_parser_output_matches_csv(self):
        """Test that parser output matches expected CSV."""
        parser = load_parser("icici")
        pdf_path = "data/icici/icici sample.pdf"
        csv_path = "data/icici/result.csv"

        result_df = parser.parse(pdf_path)
        expected_df = pd.read_csv(csv_path)

        # Normalize for comparison
        result_df = result_df.replace('', pd.NA)
        expected_df = expected_df.replace('', pd.NA)

        # Check shape
        assert result_df.shape == expected_df.shape, f"Shape mismatch: {result_df.shape} != {expected_df.shape}"

        # Check if DataFrames are equal
        assert result_df.equals(expected_df), "Parser output does not match expected CSV"

    def test_icici_parser_data_types(self):
        """Test that parser returns correct data types."""
        parser = load_parser("icici")
        pdf_path = "data/icici/icici sample.pdf"

        result_df = parser.parse(pdf_path)

        # Check that Date is string
        assert result_df["Date"].dtype == object, "Date should be string/object type"

        # Check that Description is string
        assert result_df["Description"].dtype == object, "Description should be string/object type"


def test_parser_contract():
    """Test that all parsers follow the contract."""
    parser_dir = Path("custom_parsers")
    parser_files = list(parser_dir.glob("*_parser.py"))

    assert len(parser_files) > 0, "No parsers found in custom_parsers/"

    for parser_file in parser_files:
        bank_name = parser_file.stem.replace("_parser", "")
        parser = load_parser(bank_name)

        # Test parse function exists
        assert hasattr(parser, "parse"), f"{bank_name} parser missing parse() function"

        # Test parse function signature
        import inspect
        sig = inspect.signature(parser.parse)
        assert len(sig.parameters) == 1, f"{bank_name} parse() should take 1 parameter (pdf_path)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
