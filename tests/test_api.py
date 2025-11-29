#!/usr/bin/env python3
"""
AI Inference API Test Suite
============================

This script tests the FastAPI-based AI inference service running in Docker.
It validates object detection endpoints with YOLOv8 models.

Usage:
    # Test against local Docker container
    python test_api.py --host localhost --port 8001

    # Test against remote server
    python test_api.py --host 192.168.1.100 --port 8001

    # Test with specific image
    python test_api.py --image test_images/dog_bike_car.jpg

    # Run all tests
    python test_api.py --all
"""

import os
import sys
import json
import argparse
import requests
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()

class APITester:
    """Test suite for AI Inference API"""
    
    def __init__(self, host: str = "localhost", port: int = 8001):
        self.base_url = f"http://{host}:{port}"
        self.test_images_dir = Path("test_images")
        self.results = []
        
    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make HTTP request and return response"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                "elapsed": response.elapsed.total_seconds()
            }
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Connection refused - is the API running?"}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_health(self) -> bool:
        """Test health check endpoint"""
        console.print("\n[bold blue]Testing: Health Check[/bold blue]")
        
        result = self._request("GET", "/health")
        
        if result.get("success"):
            console.print(f"[green]✅ Health check passed[/green]")
            console.print(f"   Status: {result['data'].get('status', 'unknown')}")
            return True
        else:
            console.print(f"[red]❌ Health check failed: {result.get('error', result)}[/red]")
            return False
    
    def test_root(self) -> bool:
        """Test root endpoint"""
        console.print("\n[bold blue]Testing: Root Endpoint[/bold blue]")
        
        result = self._request("GET", "/")
        
        if result.get("success"):
            console.print(f"[green]✅ Root endpoint passed[/green]")
            console.print(f"   Message: {result['data'].get('message', 'N/A')}")
            return True
        else:
            console.print(f"[red]❌ Root endpoint failed: {result.get('error', result)}[/red]")
            return False
    
    def test_list_models(self) -> bool:
        """Test list models endpoint"""
        console.print("\n[bold blue]Testing: List Models[/bold blue]")
        
        result = self._request("GET", "/models")
        
        if result.get("success"):
            models = result['data'].get('available_models', [])
            console.print(f"[green]✅ List models passed[/green]")
            console.print(f"   Available models: {models}")
            return True
        else:
            console.print(f"[red]❌ List models failed: {result.get('error', result)}[/red]")
            return False
    
    def test_list_objects(self) -> bool:
        """Test list objects endpoint"""
        console.print("\n[bold blue]Testing: List Objects[/bold blue]")
        
        result = self._request("GET", "/objects")
        
        if result.get("success"):
            objects = result['data'].get('available_objects', [])
            console.print(f"[green]✅ List objects passed[/green]")
            console.print(f"   Detectable objects: {len(objects)} types")
            if objects:
                console.print(f"   Sample: {objects[:5]}...")
            return True
        else:
            console.print(f"[red]❌ List objects failed: {result.get('error', result)}[/red]")
            return False
    
    def test_model_info(self) -> bool:
        """Test model info endpoint"""
        console.print("\n[bold blue]Testing: Model Info[/bold blue]")
        
        result = self._request("GET", "/model/info")
        
        if result.get("success"):
            data = result['data']
            console.print(f"[green]✅ Model info passed[/green]")
            console.print(f"   Architecture: {data.get('architecture', 'N/A')}")
            console.print(f"   Model types: {data.get('model_types', [])}")
            console.print(f"   Accelerators: {data.get('accelerators', [])}")
            return True
        else:
            console.print(f"[red]❌ Model info failed: {result.get('error', result)}[/red]")
            return False
    
    def test_load_model(self, model_name: str = "yolov8n") -> bool:
        """Test model load endpoint"""
        console.print(f"\n[bold blue]Testing: Load Model ({model_name})[/bold blue]")
        
        result = self._request("POST", f"/model/load?model_name={model_name}&accelerator=cpu32")
        
        if result.get("success"):
            console.print(f"[green]✅ Model load passed[/green]")
            console.print(f"   Model: {result['data'].get('model_name', 'N/A')}")
            console.print(f"   Status: {result['data'].get('status', 'N/A')}")
            return True
        else:
            console.print(f"[red]❌ Model load failed: {result.get('error', result.get('data', result))}[/red]")
            return False
    
    def test_detection(self, image_path: str, object_name: str = "car") -> bool:
        """Test object detection endpoint"""
        console.print(f"\n[bold blue]Testing: Object Detection[/bold blue]")
        console.print(f"   Image: {image_path}")
        console.print(f"   Object: {object_name}")
        
        if not os.path.exists(image_path):
            console.print(f"[red]❌ Image not found: {image_path}[/red]")
            return False
        
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
            result = self._request("POST", f"/inference/detection?object_name={object_name}", files=files)
        
        if result.get("success"):
            detections = result['data'].get('objects', [])
            console.print(f"[green]✅ Detection passed[/green]")
            console.print(f"   Detections: {len(detections)} objects found")
            console.print(f"   Inference time: {result.get('elapsed', 0):.3f}s")
            
            for i, det in enumerate(detections[:5]):
                console.print(f"   [{i+1}] class_id={det.get('class_id')}, conf={det.get('confidence', 0):.3f}")
            
            return True
        else:
            console.print(f"[red]❌ Detection failed: {result.get('error', result.get('data', result))}[/red]")
            return False
    
    def test_direct_inference(self, image_path: str, model_name: str = "yolov8n") -> bool:
        """Test direct inference endpoint"""
        console.print(f"\n[bold blue]Testing: Direct Inference[/bold blue]")
        console.print(f"   Image: {image_path}")
        console.print(f"   Model: {model_name}")
        
        if not os.path.exists(image_path):
            console.print(f"[red]❌ Image not found: {image_path}[/red]")
            return False
        
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
            result = self._request("POST", f"/inference/direct?model_name={model_name}", files=files)
        
        if result.get("success"):
            data = result['data']
            console.print(f"[green]✅ Direct inference passed[/green]")
            console.print(f"   Model: {data.get('model_name', 'N/A')}")
            console.print(f"   Input shape: {data.get('input_shape', 'N/A')}")
            console.print(f"   Output shape: {data.get('output_shape', 'N/A')}")
            console.print(f"   Inference time: {result.get('elapsed', 0):.3f}s")
            return True
        else:
            console.print(f"[red]❌ Direct inference failed: {result.get('error', result.get('data', result))}[/red]")
            return False
    
    def test_model_capabilities(self, model_name: str = "yolov8n") -> bool:
        """Test model capabilities endpoint"""
        console.print(f"\n[bold blue]Testing: Model Capabilities ({model_name})[/bold blue]")
        
        result = self._request("GET", f"/models/{model_name}/capabilities")
        
        if result.get("success"):
            data = result['data']
            console.print(f"[green]✅ Model capabilities passed[/green]")
            console.print(f"   Model: {data.get('model_name', 'N/A')}")
            capabilities = data.get('capabilities', {})
            if capabilities:
                console.print(f"   Classes: {capabilities.get('num_classes', 'N/A')}")
                console.print(f"   Input size: {capabilities.get('input_size', 'N/A')}")
            return True
        else:
            console.print(f"[red]❌ Model capabilities failed: {result.get('error', result.get('data', result))}[/red]")
            return False
    
    def benchmark_inference(self, image_path: str, model_name: str = "yolov8n", iterations: int = 10) -> dict:
        """Benchmark inference performance"""
        console.print(f"\n[bold blue]Benchmarking: {model_name} ({iterations} iterations)[/bold blue]")
        
        if not os.path.exists(image_path):
            console.print(f"[red]❌ Image not found: {image_path}[/red]")
            return {}
        
        times = []
        
        with Progress() as progress:
            task = progress.add_task("Running benchmark...", total=iterations)
            
            for i in range(iterations):
                with open(image_path, 'rb') as f:
                    files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
                    start = time.time()
                    result = self._request("POST", f"/inference/direct?model_name={model_name}", files=files)
                    elapsed = time.time() - start
                    
                    if result.get("success"):
                        times.append(elapsed)
                
                progress.update(task, advance=1)
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            console.print(f"\n[bold cyan]═══ Benchmark Results ═══[/bold cyan]")
            console.print(f"[green]✅ Completed {len(times)}/{iterations} iterations[/green]")
            console.print(f"   Average time: {avg_time*1000:.2f}ms")
            console.print(f"   Min time: {min_time*1000:.2f}ms")
            console.print(f"   Max time: {max_time*1000:.2f}ms")
            console.print(f"   Throughput: {fps:.2f} FPS")
            
            return {
                "iterations": len(times),
                "avg_time_ms": avg_time * 1000,
                "min_time_ms": min_time * 1000,
                "max_time_ms": max_time * 1000,
                "fps": fps
            }
        else:
            console.print(f"[red]❌ Benchmark failed - no successful iterations[/red]")
            return {}
    
    def run_all_tests(self, image_path: str = None) -> dict:
        """Run all API tests"""
        console.print("\n[bold magenta]╔════════════════════════════════════════╗[/bold magenta]")
        console.print("[bold magenta]║     AI Inference API Test Suite        ║[/bold magenta]")
        console.print("[bold magenta]╚════════════════════════════════════════╝[/bold magenta]")
        console.print(f"\n[cyan]Target: {self.base_url}[/cyan]")
        
        results = {
            "health": self.test_health(),
            "root": self.test_root(),
            "list_models": self.test_list_models(),
            "list_objects": self.test_list_objects(),
            "model_info": self.test_model_info(),
        }
        
        # Find a test image
        if image_path and os.path.exists(image_path):
            test_image = image_path
        else:
            # Try to find a test image
            candidates = [
                "test_images/dog_bike_car.jpg",
                "test_images/bus.jpg",
                "test_images/sample.jpg",
            ]
            test_image = None
            for candidate in candidates:
                if os.path.exists(candidate):
                    test_image = candidate
                    break
        
        if test_image:
            results["load_model"] = self.test_load_model("yolov8n")
            results["detection"] = self.test_detection(test_image, "car")
            results["direct_inference"] = self.test_direct_inference(test_image, "yolov8n")
            results["model_capabilities"] = self.test_model_capabilities("yolov8n")
        else:
            console.print("\n[yellow]⚠️ No test image found, skipping detection tests[/yellow]")
            console.print("   Place an image in test_images/ or use --image flag")
        
        # Summary
        console.print("\n[bold cyan]═══ Test Summary ═══[/bold cyan]")
        
        table = Table()
        table.add_column("Test", style="cyan")
        table.add_column("Result", style="green")
        
        passed = 0
        failed = 0
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            table.add_row(test_name, status)
            if result:
                passed += 1
            else:
                failed += 1
        
        console.print(table)
        console.print(f"\n[bold]Total: {passed} passed, {failed} failed[/bold]")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="AI Inference API Test Suite")
    parser.add_argument("--host", type=str, default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8001, help="API port")
    parser.add_argument("--image", type=str, help="Test image path")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    
    args = parser.parse_args()
    
    tester = APITester(host=args.host, port=args.port)
    
    if args.benchmark:
        image = args.image or "test_images/dog_bike_car.jpg"
        tester.benchmark_inference(image, iterations=args.iterations)
    elif args.all or not args.image:
        tester.run_all_tests(args.image)
    else:
        # Just test detection with provided image
        tester.test_health()
        tester.test_detection(args.image)


if __name__ == "__main__":
    main()

