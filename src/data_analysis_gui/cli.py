#!/usr/bin/env python3
"""
Command-line interface for the MAT file analyzer.
This demonstrates that the business logic works completely independently of the GUI.
"""

import argparse
import sys
import os
from typing import List

# Import ONLY the controller - no GUI dependencies
from data_analysis_gui.core.app_controller import ApplicationController


class CLIInterface:
    """Command-line interface for the analyzer"""
    
    def __init__(self):
        self.controller = ApplicationController()
        
        # Set up callbacks for CLI output
        self.controller.on_error = self.print_error
        self.controller.on_status_update = self.print_status
    
    def print_error(self, message: str):
        """Print error to stderr"""
        print(f"ERROR: {message}", file=sys.stderr)
    
    def print_status(self, message: str):
        """Print status to stdout"""
        print(f"STATUS: {message}")
    
    def analyze_file(self, file_path: str, output_dir: str, 
                    range1_start: float = 50.0, range1_end: float = 100.0,
                    use_dual: bool = False, range2_start: float = 150.0, 
                    range2_end: float = 200.0):
        """Analyze a single file"""
        print(f"\nAnalyzing: {file_path}")
        
        # Load the file
        file_info = self.controller.load_file(file_path)
        if not file_info:
            return False
        
        print(f"  Loaded: {file_info.sweep_count} sweeps")
        
        # Build parameters
        params = self.controller.build_parameters(
            range1_start=range1_start,
            range1_end=range1_end,
            use_dual_range=use_dual,
            range2_start=range2_start if use_dual else None,
            range2_end=range2_end if use_dual else None,
            stimulus_period=500.0,
            x_measure="Average",
            x_channel="Voltage",
            y_measure="Average",
            y_channel="Current",
            channel_config=self.controller.channel_definitions.get_configuration()
        )
        
        # Perform analysis
        result = self.controller.perform_analysis(params)
        if not result:
            return False
        
        print(f"  Analysis complete: {len(result.x_data)} data points")
        
        # Export results
        success = self.controller.export_analysis_data(params, output_dir)
        if success:
            print(f"  Exported to: {output_dir}")
        
        return success
    
    def batch_analyze(self, file_paths: List[str], output_dir: str,
                     range1_start: float = 50.0, range1_end: float = 100.0,
                     use_dual: bool = False, range2_start: float = 150.0,
                     range2_end: float = 200.0):
        """Perform batch analysis"""
        print(f"\nBatch analyzing {len(file_paths)} files...")
        
        # Build parameters
        params = self.controller.build_parameters(
            range1_start=range1_start,
            range1_end=range1_end,
            use_dual_range=use_dual,
            range2_start=range2_start if use_dual else None,
            range2_end=range2_end if use_dual else None,
            stimulus_period=500.0,
            x_measure="Average",
            x_channel="Voltage",
            y_measure="Average",
            y_channel="Current",
            channel_config=self.controller.channel_definitions.get_configuration()
        )
        
        # Progress callback
        def on_progress(current, total):
            print(f"  Progress: {current}/{total}")
        
        # File complete callback
        def on_file_complete(result):
            if result.success:
                print(f"  ✓ {result.base_name}: {len(result.x_data)} points")
            else:
                print(f"  ✗ {result.base_name}: {result.error_message}")
        
        # Run batch analysis
        result = self.controller.perform_batch_analysis(
            file_paths,
            params,
            output_dir,
            on_progress=on_progress,
            on_file_complete=on_file_complete
        )
        
        if result['success']:
            success_count = sum(1 for r in result['results'] if r.success)
            print(f"\nBatch complete: {success_count}/{len(file_paths)} files processed")
            print(f"Results saved to: {output_dir}")
            return True
        else:
            print(f"\nBatch failed: {result.get('error', 'Unknown error')}")
            return False
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("\nMAT File Analyzer - Interactive Mode")
        print("=" * 40)
        
        while True:
            print("\nOptions:")
            print("1. Load and analyze a file")
            print("2. Batch analyze files")
            print("3. Swap channels")
            print("4. Show current configuration")
            print("5. Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                file_path = input("Enter MAT file path: ").strip()
                if not os.path.exists(file_path):
                    print("File not found!")
                    continue
                
                output_dir = input("Enter output directory: ").strip()
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                use_dual = input("Use dual range? (y/n): ").lower() == 'y'
                
                self.analyze_file(file_path, output_dir, use_dual=use_dual)
            
            elif choice == '2':
                file_pattern = input("Enter directory or file pattern: ").strip()
                output_dir = input("Enter output directory: ").strip()
                
                if os.path.isdir(file_pattern):
                    import glob
                    files = glob.glob(os.path.join(file_pattern, "*.mat"))
                else:
                    import glob
                    files = glob.glob(file_pattern)
                
                if not files:
                    print("No files found!")
                    continue
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                use_dual = input("Use dual range? (y/n): ").lower() == 'y'
                
                self.batch_analyze(files, output_dir, use_dual=use_dual)
            
            elif choice == '3':
                result = self.controller.swap_channels()
                if result['success']:
                    status = "swapped" if result['is_swapped'] else "normal"
                    print(f"Channels {status}: {result['configuration']}")
                else:
                    print(f"Cannot swap: {result.get('reason', 'Unknown reason')}")
            
            elif choice == '4':
                config = self.controller.channel_definitions.get_configuration()
                print(f"Current configuration: {config}")
                print(f"Available types: {self.controller.get_channel_types()}")
            
            elif choice == '5':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice!")


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(description='MAT File Analyzer CLI')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--batch', '-b', nargs='+', 
                       help='Batch analyze multiple files')
    parser.add_argument('--file', '-f', 
                       help='Single file to analyze')
    parser.add_argument('--output', '-o', default='./output',
                       help='Output directory (default: ./output)')
    parser.add_argument('--range1-start', type=float, default=50.0,
                       help='Range 1 start time in ms')
    parser.add_argument('--range1-end', type=float, default=100.0,
                       help='Range 1 end time in ms')
    parser.add_argument('--dual-range', action='store_true',
                       help='Use dual range analysis')
    parser.add_argument('--range2-start', type=float, default=150.0,
                       help='Range 2 start time in ms')
    parser.add_argument('--range2-end', type=float, default=200.0,
                       help='Range 2 end time in ms')
    
    args = parser.parse_args()
    
    cli = CLIInterface()
    
    if args.interactive:
        cli.interactive_mode()
    elif args.batch:
        # Create output directory if needed
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        
        success = cli.batch_analyze(
            args.batch, args.output,
            range1_start=args.range1_start,
            range1_end=args.range1_end,
            use_dual=args.dual_range,
            range2_start=args.range2_start,
            range2_end=args.range2_end
        )
        sys.exit(0 if success else 1)
    elif args.file:
        # Create output directory if needed
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        
        success = cli.analyze_file(
            args.file, args.output,
            range1_start=args.range1_start,
            range1_end=args.range1_end,
            use_dual=args.dual_range,
            range2_start=args.range2_start,
            range2_end=args.range2_end
        )
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()