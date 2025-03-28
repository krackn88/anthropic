"""
System utility tools for the Anthropic-powered Agent
"""

import os
import json
import shlex
import asyncio
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_system_tools() -> List:
    """Get tools for system operations."""
    from anthropic_agent import Tool, ToolParameter
    
    async def read_file(path: str) -> Dict[str, Any]:
        """Read the contents of a file."""
        try:
            # Normalize path
            path = os.path.abspath(os.path.expanduser(path))
            
            # Check if file exists
            if not os.path.isfile(path):
                return {"error": f"File not found: {path}"}
            
            # Check file size
            if os.path.getsize(path) > 10 * 1024 * 1024:  # 10MB limit
                return {"error": f"File is too large (>10MB): {path}"}
            
            # Read the file
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            return {
                "content": content,
                "path": path,
                "size_bytes": os.path.getsize(path),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
            }
        except UnicodeDecodeError:
            return {"error": f"File is not a text file or has an unsupported encoding: {path}"}
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return {"error": str(e)}
    
    async def write_file(path: str, content: str) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            # Normalize path
            path = os.path.abspath(os.path.expanduser(path))
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Write to the file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return {
                "status": "success",
                "path": path,
                "size_bytes": os.path.getsize(path),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
            }
        except Exception as e:
            logger.error(f"Error writing to file: {str(e)}")
            return {"error": str(e)}
    
    async def execute_command(command: str) -> Dict[str, Any]:
        """Execute a system command."""
        try:
            # Only allow certain safe commands
            allowed_commands = ["ls", "dir", "cd", "pwd", "echo", "cat", "type", "grep", "find", "wc"]
            
            # Parse command
            args = shlex.split(command)
            if not args:
                return {"error": "Empty command"}
            
            base_command = os.path.basename(args[0])
            
            if base_command not in allowed_commands:
                return {"error": f"Command not allowed: {base_command}. Allowed commands: {', '.join(allowed_commands)}"}
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "stdout": stdout.decode("utf-8"),
                "stderr": stderr.decode("utf-8"),
                "returncode": process.returncode
            }
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            return {"error": str(e)}
    
    async def analyze_csv(file_path: str) -> Dict[str, Any]:
        """Analyze a CSV file and return summary statistics."""
        try:
            # Normalize path
            file_path = os.path.abspath(os.path.expanduser(file_path))
            
            # Check if file exists
            if not os.path.isfile(file_path):
                return {"error": f"File not found: {file_path}"}
            
            # Check file extension
            if not file_path.lower().endswith('.csv'):
                return {"error": f"File is not a CSV file: {file_path}"}
            
            # Read CSV file
            data = pd.read_csv(file_path)
            
            # Generate summary
            summary = {
                "filename": os.path.basename(file_path),
                "shape": {"rows": data.shape[0], "columns": data.shape[1]},
                "columns": list(data.columns),
                "column_types": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "missing_values": {col: int(data[col].isna().sum()) for col in data.columns},
                "summary_statistics": {}
            }
            
            # Add summary statistics for numeric columns
            numeric_columns = data.select_dtypes(include=['number']).columns
            if not numeric_columns.empty:
                stats = data[numeric_columns].describe().to_dict()
                summary["summary_statistics"] = stats
            
            # Add value counts for categorical columns (top 5 values)
            categorical_columns = data.select_dtypes(include=['object']).columns
            if not categorical_columns.empty:
                summary["categorical_values"] = {}
                for col in categorical_columns:
                    value_counts = data[col].value_counts().head(5).to_dict()
                    summary["categorical_values"][col] = value_counts
            
            return summary
        except Exception as e:
            logger.error(f"Error analyzing CSV file: {str(e)}")
            return {"error": str(e)}
    
    async def plot_data(data: str, x: str, y: str, plot_type: str = "line", title: str = None, output_path: str = "plot.png") -> Dict[str, Any]:
        """Plot data from a JSON string and save the plot to a file."""
        try:
            # Parse data
            try:
                if isinstance(data, str):
                    df = pd.read_json(data)
                else:
                    return {"error": "Data must be a JSON string"}
            except Exception as e:
                return {"error": f"Error parsing JSON data: {str(e)}"}
            
            # Check if columns exist
            if x not in df.columns:
                return {"error": f"Column '{x}' not found in data"}
            if y not in df.columns:
                return {"error": f"Column '{y}' not found in data"}
            
            # Normalize output path
            output_path = os.path.abspath(os.path.expanduser(output_path))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            
            if plot_type == "line":
                plt.plot(df[x], df[y])
            elif plot_type == "scatter":
                plt.scatter(df[x], df[y])
            elif plot_type == "bar":
                plt.bar(df[x], df[y])
            elif plot_type == "histogram":
                plt.hist(df[y], bins=10)
            else:
                return {"error": f"Unsupported plot type: {plot_type}"}
            
            # Add labels and title
            plt.xlabel(x)
            plt.ylabel(y)
            if title:
                plt.title(title)
            else:
                plt.title(f"{y} vs {x}")
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
            
            return {
                "status": "success",
                "output_path": output_path,
                "plot_type": plot_type,
                "x": x,
                "y": y,
                "data_shape": {"rows": df.shape[0], "columns": df.shape[1]}
            }
        except Exception as e:
            logger.error(f"Error plotting data: {str(e)}")
            return {"error": str(e)}
    
    async def list_directory(path: str = ".", pattern: Optional[str] = None) -> Dict[str, Any]:
        """List files and directories in a directory, optionally filtered by a pattern."""
        try:
            # Normalize path
            path = os.path.abspath(os.path.expanduser(path))
            
            # Check if directory exists
            if not os.path.isdir(path):
                return {"error": f"Directory not found: {path}"}
            
            # List directory contents
            if pattern:
                from glob import glob
                items = glob(os.path.join(path, pattern))
            else:
                items = os.listdir(path)
            
            # Get file/directory information
            result = []
            for item in items:
                if pattern:
                    item_path = item
                    item_name = os.path.basename(item)
                else:
                    item_path = os.path.join(path, item)
                    item_name = item
                
                is_dir = os.path.isdir(item_path)
                
                if is_dir:
                    result.append({
                        "name": item_name,
                        "type": "directory",
                        "last_modified": datetime.fromtimestamp(os.path.getmtime(item_path)).isoformat()
                    })
                else:
                    result.append({
                        "name": item_name,
                        "type": "file",
                        "size_bytes": os.path.getsize(item_path),
                        "last_modified": datetime.fromtimestamp(os.path.getmtime(item_path)).isoformat()
                    })
            
            return {
                "path": path,
                "items": result,
                "count": len(result)
            }
        except Exception as e:
            logger.error(f"Error listing directory: {str(e)}")
            return {"error": str(e)}
    
    # Define tools
    return [
        Tool(
            name="read_file",
            description="Read the contents of a file",
            parameters=[
                ToolParameter(name="path", type="string", description="Path to the file")
            ],
            function=read_file,
            category="system"
        ),
        
        Tool(
            name="write_file",
            description="Write content to a file",
            parameters=[
                ToolParameter(name="path", type="string", description="Path to the file"),
                ToolParameter(name="content", type="string", description="Content to write")
            ],
            function=write_file,
            category="system"
        ),
        
        Tool(
            name="execute_command",
            description="Execute a system command (limited to safe commands)",
            parameters=[
                ToolParameter(name="command", type="string", description="Command to execute")
            ],
            function=execute_command,
            category="system"
        ),
        
        Tool(
            name="analyze_csv",
            description="Analyze a CSV file and return summary statistics",
            parameters=[
                ToolParameter(name="file_path", type="string", description="Path to the CSV file")
            ],
            function=analyze_csv,
            category="system"
        ),
        
        Tool(
            name="plot_data",
            description="Plot data from a JSON string and save the plot to a file",
            parameters=[
                ToolParameter(name="data", type="string", description="JSON string of the data"),
                ToolParameter(name="x", type="string", description="Column name for x-axis"),
                ToolParameter(name="y", type="string", description="Column name for y-axis"),
                ToolParameter(name="plot_type", type="string", description="Type of plot (line, scatter, bar, histogram)", required=False, default="line"),
                ToolParameter(name="title", type="string", description="Plot title", required=False),
                ToolParameter(name="output_path", type="string", description="Path to save the plot", required=False, default="plot.png")
            ],
            function=plot_data,
            category="system"
        ),
        
        Tool(
            name="list_directory",
            description="List files and directories in a directory, optionally filtered by a pattern",
            parameters=[
                ToolParameter(name="path", type="string", description="Directory path", required=False, default="."),
                ToolParameter(name="pattern", type="string", description="Filter pattern (e.g., '*.py')", required=False)
            ],
            function=list_directory,
            category="system"
        )
    ]