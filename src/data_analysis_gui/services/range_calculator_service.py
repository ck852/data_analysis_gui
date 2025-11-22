"""
PatchBatch Electrophysiology Data Analysis Tool

Range Calculator Service - extends concentration-response analysis with 
custom equation evaluation across user-defined range variables.

Author: Charles Kissell, Northeastern University  
License: MIT (see LICENSE file for details)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re

from data_analysis_gui.core.conc_resp_models import AnalysisRange, AnalysisType
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class RangeCalculatorService:
    """
    Service for defining custom equations using range-based variables.
    
    Extends ConcentrationResponseService by allowing users to:
    1. Assign variable names to AnalysisRange objects
    2. Define a custom equation using those variables
    3. Calculate equation results across all data traces
    
    Example workflow:
        >>> calc = RangeCalculatorService()
        >>> calc.assign_variable('x', 'Range_1')  # baseline
        >>> calc.assign_variable('p', 'Range_2')  # PLL
        >>> calc.assign_variable('d', 'Range_3')  # diC8
        >>> calc.set_equation('100 * (d - p) / (x - p)')
        >>> results = calc.calculate_for_traces(df, time_col, data_cols, ranges)
    """
    
    def __init__(self):
        """Initialize calculator with empty variable mappings and equation."""
        self.variable_map: Dict[str, str] = {}  # {var_name: range_id}
        self.equation: str = ""
        self._validated_vars: set = set()
    
    def assign_variable(self, var_name: str, range_id: str) -> None:
        """
        Assign a variable name to a range ID.
        
        Args:
            var_name: Variable name (e.g., 'x', 'baseline', 'pll')
            range_id: Range identifier (e.g., 'Range_1')
        
        Raises:
            ValueError: If var_name contains invalid characters
        """
        # Validate variable name (letters, numbers, underscore only)
        if not re.match(r'^[a-zA-Z_]\w*$', var_name):
            raise ValueError(
                f"Invalid variable name '{var_name}'. Must start with letter/underscore "
                f"and contain only letters, numbers, and underscores."
            )
        
        self.variable_map[var_name] = range_id
        logger.debug(f"Assigned variable '{var_name}' to range '{range_id}'")
    
    def remove_variable(self, var_name: str) -> None:
        """Remove a variable assignment."""
        if var_name in self.variable_map:
            del self.variable_map[var_name]
            logger.debug(f"Removed variable assignment for '{var_name}'")
    
    def clear_variables(self) -> None:
        """Clear all variable assignments."""
        self.variable_map.clear()
        logger.debug("Cleared all variable assignments")
    
    def set_equation(self, equation: str) -> Tuple[bool, str]:
        """
        Set the equation to calculate.
        
        Validates that all variables in the equation have been assigned
        to ranges (except for allowed math functions).
        
        Args:
            equation: String equation (e.g., "100 * (d - p) / (x - p)")
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not equation.strip():
            return False, "Equation cannot be empty"
        
        # Extract variables from equation
        variables_in_eq = set(re.findall(r'\b[a-zA-Z_]\w*\b', equation))
        
        # Remove allowed functions
        allowed_funcs = {'abs', 'max', 'min', 'sqrt', 'exp', 'log', 'log10', 'np'}
        unknown_vars = variables_in_eq - set(self.variable_map.keys()) - allowed_funcs
        
        if unknown_vars:
            return False, f"Undefined variables in equation: {', '.join(sorted(unknown_vars))}"
        
        self.equation = equation
        self._validated_vars = variables_in_eq & set(self.variable_map.keys())
        
        logger.info(f"Set equation: {equation}")
        return True, "Equation validated successfully"
    
    def get_required_ranges(self) -> List[str]:
        """
        Get list of range IDs required for current equation.
        
        Returns:
            List of range_id values needed to evaluate the equation
        """
        if not self.equation or not self._validated_vars:
            return []
        
        return [self.variable_map[var] for var in self._validated_vars]
    
    def _extract_range_values(
        self,
        df: pd.DataFrame,
        time_col: str,
        data_col: str,
        ranges: List[AnalysisRange],  # Updated type
        statistic: str = 'mean'
    ) -> Dict[str, float]:
        """
        Extract values for all assigned variables from a single data trace.
        
        Args:
            df: DataFrame containing time-series data
            time_col: Name of time column
            data_col: Name of data column to analyze
            ranges: List of AnalysisRange objects
            statistic: 'mean', 'median', 'max', 'min', or 'last'
        
        Returns:
            Dictionary mapping variable names to extracted values
        """
        # Build lookup of range_id -> AnalysisRange
        range_lookup = {r.range_id: r for r in ranges}
        
        # Extract values for each variable
        values = {}
        for var_name, range_id in self.variable_map.items():
            if range_id not in range_lookup:
                logger.warning(f"Range '{range_id}' not found for variable '{var_name}'")
                values[var_name] = np.nan
                continue
            
            range_obj = range_lookup[range_id]
            
            # Get data subset
            mask = (df[time_col] >= range_obj.start_time) & (df[time_col] <= range_obj.end_time)
            subset = df.loc[mask, data_col]
            
            if subset.empty:
                logger.warning(
                    f"No data in range [{range_obj.start_time}, {range_obj.end_time}] "
                    f"for '{data_col}'"
                )
                values[var_name] = np.nan
                continue
            
            # Calculate statistic
            if statistic == 'mean':
                values[var_name] = float(subset.mean())
            elif statistic == 'median':
                values[var_name] = float(subset.median())
            elif statistic == 'max':
                values[var_name] = float(subset.max())
            elif statistic == 'min':
                values[var_name] = float(subset.min())
            elif statistic == 'last':
                values[var_name] = float(subset.iloc[-1])
            else:
                raise ValueError(f"Unknown statistic: {statistic}")
        
        return values
    
    def _evaluate_equation(self, values: Dict[str, float]) -> Optional[float]:
        """
        Safely evaluate the equation with given variable values.
        
        Args:
            values: Dictionary mapping variable names to values
        
        Returns:
            Calculated result, or None if evaluation fails
        """
        if not self.equation:
            return None
        
        try:
            # Create safe evaluation context
            eval_context = {
                **values,
                'abs': abs,
                'max': max,
                'min': min,
                'sqrt': np.sqrt,
                'exp': np.exp,
                'log': np.log,
                'log10': np.log10,
                'np': np
            }
            
            result = eval(self.equation, {"__builtins__": {}}, eval_context)
            return float(result)
            
        except Exception as e:
            logger.error(f"Error evaluating equation '{self.equation}': {e}")
            return None
    
    def calculate_for_traces(
        self,
        df: pd.DataFrame,
        time_col: str,
        data_cols: List[str],
        ranges: List[AnalysisRange],  # Updated type
        filename: str = "data",
        statistic: str = 'mean'
    ) -> pd.DataFrame:
        """
        Calculate equation results for all data traces.
        
        Args:
            df: DataFrame containing time-series data
            time_col: Name of time column
            data_cols: List of data column names to analyze
            ranges: List of AnalysisRange objects
            filename: Source filename (for results table)
            statistic: Statistical measure to extract from ranges
        
        Returns:
            DataFrame with columns:
                - File: source filename
                - Data Trace: name of data column
                - {var_name}: value for each variable
                - Result: calculated equation result
        
        Raises:
            ValueError: If equation not set or variables not assigned
        """
        if not self.equation:
            raise ValueError("No equation set. Call set_equation() first.")
        
        if not self.variable_map:
            raise ValueError("No variables assigned. Call assign_variable() first.")
        
        results_rows = []
        
        for data_col in data_cols:
            # Extract all variable values for this trace
            values = self._extract_range_values(
                df, time_col, data_col, ranges, statistic
            )
            
            # Evaluate equation
            result = self._evaluate_equation(values)
            
            # Build result row
            row = {
                'File': filename,
                'Data Trace': data_col,
                **values,
                'Result': result
            }
            results_rows.append(row)
        
        results_df = pd.DataFrame(results_rows)
        
        logger.info(
            f"Calculated equation for {len(data_cols)} trace(s): "
            f"{len(results_rows)} result(s)"
        )
        
        return results_df
    
    def get_summary(self) -> str:
        """
        Get human-readable summary of calculator configuration.
        
        Returns:
            Multi-line string describing variables and equation
        """
        lines = ["Range Calculator Configuration:"]
        lines.append(f"  Variables: {len(self.variable_map)}")
        
        for var_name, range_id in sorted(self.variable_map.items()):
            lines.append(f"    {var_name} → {range_id}")
        
        if self.equation:
            lines.append(f"  Equation: {self.equation}")
        else:
            lines.append("  Equation: (not set)")
        
        return "\n".join(lines)


# Example integration with existing service
class ConcentrationResponseServiceExtended:
    """
    Extended service combining standard analysis with calculator.
    
    This shows how to integrate RangeCalculatorService with the 
    existing ConcentrationResponseService workflow.
    """
    
    def __init__(self):
        from data_analysis_gui.services.conc_resp_service import ConcentrationResponseService
        self.standard_service = ConcentrationResponseService()
        self.calculator = RangeCalculatorService()
    
    def run_analysis_with_calculator(
        self,
        df: pd.DataFrame,
        time_col: str,
        data_cols: List[str],
        ranges: List[AnalysisRange],  # Updated type
        filename: str = "data",
        use_calculator: bool = False,
        statistic: str = 'mean'
    ) -> Dict[str, pd.DataFrame]:
        """
        Run analysis with optional custom equation calculation.
        
        Args:
            df: DataFrame with time-series data
            time_col: Time column name
            data_cols: Data column names
            ranges: AnalysisRange objects
            filename: Source file name
            use_calculator: If True, use calculator equation; else standard analysis
            statistic: Statistic for range extraction (calculator only)
        
        Returns:
            Dict mapping trace names to results DataFrames
        """
        if use_calculator and self.calculator.equation:
            # Use calculator service
            results_df = self.calculator.calculate_for_traces(
                df, time_col, data_cols, ranges, filename, statistic
            )
            # Return in same format as standard analysis
            return {col: results_df[results_df['Data Trace'] == col] 
                    for col in data_cols}
        else:
            # Use standard analysis
            return self.standard_service.run_analysis(
                df, time_col, data_cols, ranges, filename
            )