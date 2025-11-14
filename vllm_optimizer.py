#!/usr/bin/env python3
"""
vLLM GPU Optimizer
NVIDIA GPU configuration optimizer for vLLM workloads
"""

import subprocess
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

class VLLMOptimizer:
    def __init__(self):
        self.profiles = {
            'max_throughput': {
                'name': 'Maximum Throughput',
                'description': 'Optimize GPU for highest token/s throughput',
                'persistence_mode': True,
                'compute_mode': 0,  # Default (allows multiple processes)
                'power_limit': 'max',  # Use maximum available
                'auto_boost': True,
                'gpu_clocks': 'prefer_memory',  # Prioritize memory bandwidth
                'ecc': False,  # Disable ECC for more memory/performance
            },
            'low_latency': {
                'name': 'Low Latency',
                'description': 'Optimize GPU for minimal first-token latency',
                'persistence_mode': True,
                'compute_mode': 3,  # Exclusive Process
                'power_limit': 'max',
                'auto_boost': True,
                'gpu_clocks': 'prefer_compute',  # Prioritize compute speed
                'ecc': False,
            },
            'balanced': {
                'name': 'Balanced',
                'description': 'Balance between throughput and latency',
                'persistence_mode': True,
                'compute_mode': 0,
                'power_limit': 'default',
                'auto_boost': True,
                'gpu_clocks': 'balanced',
                'ecc': False,
            },
            'power_efficient': {
                'name': 'Power Efficient',
                'description': 'Reduce power consumption while maintaining good performance',
                'persistence_mode': True,
                'compute_mode': 0,
                'power_limit': 75,  # 75% of max
                'auto_boost': False,
                'gpu_clocks': 'efficient',
                'ecc': False,
            },
            'production': {
                'name': 'Production Ready',
                'description': 'Stable settings for production deployments',
                'persistence_mode': True,
                'compute_mode': 0,
                'power_limit': 90,  # 90% for thermal headroom
                'auto_boost': True,
                'gpu_clocks': 'stable',
                'ecc': True,  # Enable ECC for reliability
            }
        }
        
        self.settings_file = Path.home() / '.config' / 'nvidiacp' / 'vllm_settings.json'
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
    
    def run_command(self, cmd: List[str], check: bool = True) -> Tuple[bool, str]:
        """Run a command and return success status and output"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr
        except Exception as e:
            return False, str(e)

    def get_gpu_count(self) -> int:
        """Get number of NVIDIA GPUs"""
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        return len([l for l in result.stdout.split('\n') if l.startswith('GPU')])

    def validate_gpu_id(self, gpu_id: int) -> bool:
        """Validate that GPU ID exists"""
        gpu_count = self.get_gpu_count()
        if gpu_id < 0 or gpu_id >= gpu_count:
            print(f"✗ Invalid GPU ID: {gpu_id}. Valid range is 0-{gpu_count - 1}")
            return False
        return True
    
    def get_gpu_info(self, gpu_id: int = 0) -> Dict:
        """Get GPU information"""
        info = {}
        
        # Get GPU model and memory
        cmd = ['nvidia-smi', '-i', str(gpu_id), '--query-gpu=name,memory.total,power.limit,power.max_limit', 
               '--format=csv,noheader,nounits']
        success, output = self.run_command(cmd)
        if success and output.strip():
            parts = output.strip().split(', ')
            if len(parts) >= 4:
                info['model'] = parts[0]
                info['memory_mb'] = int(float(parts[1]))
                info['current_power_limit'] = int(float(parts[2]))
                info['max_power_limit'] = int(float(parts[3]))
        
        # Get current settings
        cmd = ['nvidia-smi', '-i', str(gpu_id), 
               '--query-gpu=persistence_mode,compute_mode,clocks.gr,clocks.mem', 
               '--format=csv,noheader']
        success, output = self.run_command(cmd)
        if success and output.strip():
            parts = output.strip().split(', ')
            if len(parts) >= 4:
                info['persistence'] = parts[0]
                info['compute_mode'] = parts[1]
                info['graphics_clock'] = parts[2]
                info['memory_clock'] = parts[3]
        
        return info
    
    def get_optimal_clocks(self, gpu_id: int, profile: str) -> Tuple[Optional[int], Optional[int]]:
        """Get optimal clock settings based on GPU and profile"""
        # Query supported clocks
        cmd = ['nvidia-smi', '-i', str(gpu_id), '-q', '-d', 'SUPPORTED_CLOCKS']
        success, output = self.run_command(cmd)
        
        mem_clocks = []
        graphics_clocks = {}
        
        if success:
            lines = output.split('\n')
            current_mem = None
            
            for line in lines:
                line = line.strip()
                if 'Memory' in line and 'MHz' in line:
                    try:
                        current_mem = int(line.split()[2])
                        mem_clocks.append(current_mem)
                        graphics_clocks[current_mem] = []
                    except:
                        pass
                elif 'Graphics' in line and 'MHz' in line and current_mem:
                    try:
                        graphics = int(line.split()[2])
                        graphics_clocks[current_mem].append(graphics)
                    except:
                        pass
        
        if not mem_clocks:
            return None, None
        
        profile_data = self.profiles[profile]
        
        # Select clocks based on profile preference
        if profile_data['gpu_clocks'] == 'prefer_memory':
            # Highest memory clock
            mem_clock = max(mem_clocks)
            graphics_clock = max(graphics_clocks.get(mem_clock, [0]))
        elif profile_data['gpu_clocks'] == 'prefer_compute':
            # Find combination with highest graphics clock
            best_graphics = 0
            best_mem = mem_clocks[0]
            for mem in mem_clocks:
                if graphics_clocks.get(mem):
                    max_graphics = max(graphics_clocks[mem])
                    if max_graphics > best_graphics:
                        best_graphics = max_graphics
                        best_mem = mem
            mem_clock = best_mem
            graphics_clock = best_graphics
        elif profile_data['gpu_clocks'] == 'efficient':
            # Middle-range clocks for efficiency
            mem_clock = mem_clocks[len(mem_clocks)//2] if mem_clocks else mem_clocks[0]
            graphics_clock = graphics_clocks[mem_clock][len(graphics_clocks[mem_clock])//2] if graphics_clocks.get(mem_clock) else 0
        elif profile_data['gpu_clocks'] == 'stable':
            # 80% of maximum for stability
            mem_clock = mem_clocks[int(len(mem_clocks) * 0.8)] if len(mem_clocks) > 1 else mem_clocks[0]
            gc_list = graphics_clocks.get(mem_clock, [])
            graphics_clock = gc_list[int(len(gc_list) * 0.8)] if len(gc_list) > 1 else gc_list[0] if gc_list else 0
        else:  # balanced
            # 90% of maximum
            mem_clock = mem_clocks[int(len(mem_clocks) * 0.9)] if len(mem_clocks) > 1 else mem_clocks[0]
            gc_list = graphics_clocks.get(mem_clock, [])
            graphics_clock = gc_list[int(len(gc_list) * 0.9)] if len(gc_list) > 1 else gc_list[0] if gc_list else 0
        
        return mem_clock, graphics_clock
    
    def apply_profile(self, gpu_id: int, profile: str) -> Dict:
        """Apply vLLM optimization profile to GPU"""
        if profile not in self.profiles:
            return {'success': False, 'message': f"Unknown profile: {profile}"}

        if not self.validate_gpu_id(gpu_id):
            return {'success': False, 'message': f"Invalid GPU ID: {gpu_id}"}

        profile_data = self.profiles[profile]
        gpu_info = self.get_gpu_info(gpu_id)
        
        print(f"\nApplying '{profile_data['name']}' profile to GPU {gpu_id}")
        print(f"Description: {profile_data['description']}")
        print("=" * 60)
        
        results = []
        
        # 1. Set persistence mode
        mode = '1' if profile_data['persistence_mode'] else '0'
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pm', mode]
        success, output = self.run_command(cmd)
        status = "✓" if success else "✗"
        results.append(f"{status} Persistence mode: {'Enabled' if profile_data['persistence_mode'] else 'Disabled'}")
        
        # 2. Set compute mode
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-c', str(profile_data['compute_mode'])]
        success, output = self.run_command(cmd)
        status = "✓" if success else "✗"
        compute_modes = {0: 'Default', 1: 'Exclusive Thread', 2: 'Prohibited', 3: 'Exclusive Process'}
        results.append(f"{status} Compute mode: {compute_modes[profile_data['compute_mode']]}")
        
        # 3. Set power limit
        max_power = gpu_info.get('max_power_limit', 300)
        if profile_data['power_limit'] == 'max':
            power_limit = max_power
        elif profile_data['power_limit'] == 'default':
            power_limit = int(max_power * 0.9)
        else:
            power_limit = int(max_power * profile_data['power_limit'] / 100)
        
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pl', str(power_limit)]
        success, output = self.run_command(cmd)
        status = "✓" if success else "✗"
        results.append(f"{status} Power limit: {power_limit}W")
        
        # 4. Set auto boost (deprecated feature, may not work on modern GPUs)
        if profile_data.get('auto_boost') is not None:
            mode = '1' if profile_data['auto_boost'] else '0'
            # NOTE: --auto-boost-default is deprecated, controls whether auto-boost is enabled
            cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '--auto-boost-default=' + mode]
            success, output = self.run_command(cmd, check=False)  # May not be supported
            if success:
                results.append(f"✓ Auto boost: {'Enabled' if profile_data['auto_boost'] else 'Disabled'} (deprecated feature)")
            else:
                results.append(f"⚠ Auto boost not supported (normal for modern GPUs)")
        
        # 5. Set GPU clocks
        mem_clock, graphics_clock = self.get_optimal_clocks(gpu_id, profile)
        if mem_clock and graphics_clock:
            cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-ac', f'{mem_clock},{graphics_clock}']
            success, output = self.run_command(cmd, check=False)
            if success:
                results.append(f"✓ Application clocks: {mem_clock}MHz mem, {graphics_clock}MHz graphics")
            else:
                # Application clocks not supported on this GPU
                results.append(f"⚠ Application clocks not supported on this GPU")
        
        # 6. Set ECC if specified
        if 'ecc' in profile_data:
            mode = '1' if profile_data['ecc'] else '0'
            cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-e', mode]
            success, output = self.run_command(cmd, check=False)
            if success:
                results.append(f"✓ ECC memory: {'Enabled (requires reboot)' if profile_data['ecc'] else 'Disabled (requires reboot)'}")
        
        # Display results
        print("\nConfiguration Results:")
        for result in results:
            print(f"  {result}")
        
        # Save applied settings
        self.save_profile_settings(gpu_id, profile, gpu_info)
        
        # Show recommendations
        self.show_vllm_recommendations(profile, gpu_info)
        
        return {'success': True, 'results': results}
    
    def save_profile_settings(self, gpu_id: int, profile: str, gpu_info: Dict):
        """Save applied profile settings"""
        settings = {
            'gpu_id': gpu_id,
            'profile': profile,
            'gpu_model': gpu_info.get('model', 'Unknown'),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'profile_data': self.profiles[profile]
        }
        
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        
        print(f"\n✓ Settings saved to: {self.settings_file}")
    
    def show_vllm_recommendations(self, profile: str, gpu_info: Dict):
        """Show vLLM-specific recommendations based on profile and GPU"""
        print("\n" + "=" * 60)
        print("vLLM Performance Recommendations:")
        print("=" * 60)
        
        model = gpu_info.get('model', '')
        memory_mb = gpu_info.get('memory_mb', 24000)
        memory_gb = memory_mb / 1024
        
        recommendations = []
        
        # Memory utilization recommendations
        if profile == 'max_throughput':
            recommendations.append(f"--gpu-memory-utilization 0.95  # Maximum memory for batching")
        elif profile == 'low_latency':
            recommendations.append(f"--gpu-memory-utilization 0.85  # Reserve memory for fast allocation")
        elif profile == 'power_efficient':
            recommendations.append(f"--gpu-memory-utilization 0.80  # Conservative memory usage")
        else:
            recommendations.append(f"--gpu-memory-utilization 0.90  # Balanced memory usage")
        
        # Model and batch size recommendations based on GPU
        if 'A100' in model or 'H100' in model:
            if memory_gb >= 80:
                recommendations.append("--max-model-len 32768      # Large context for 80GB")
                recommendations.append("--max-num-seqs 256         # High batch size")
            else:
                recommendations.append("--max-model-len 16384      # Medium context for 40GB")
                recommendations.append("--max-num-seqs 128         # Medium batch size")
        elif 'A6000' in model or 'RTX 6000' in model or 'A40' in model:
            recommendations.append("--max-model-len 16384      # Good for 48GB VRAM")
            recommendations.append("--max-num-seqs 128         # Professional GPU batch size")
        elif '4090' in model or '4080' in model:
            recommendations.append("--max-model-len 8192       # Consumer high-end GPU")
            recommendations.append("--max-num-seqs 64          # RTX 40 series batch size")
        elif '3090' in model or 'A5000' in model:
            recommendations.append("--max-model-len 8192       # 24GB VRAM limit")
            recommendations.append("--max-num-seqs 32          # Conservative batching")
        else:
            # Generic recommendations
            if memory_gb >= 24:
                recommendations.append("--max-model-len 8192       # Standard for 24GB+")
                recommendations.append("--max-num-seqs 32          # Standard batch size")
            elif memory_gb >= 16:
                recommendations.append("--max-model-len 4096       # Limited by 16GB VRAM")
                recommendations.append("--max-num-seqs 16          # Small batch size")
            else:
                recommendations.append("--max-model-len 2048       # Small VRAM")
                recommendations.append("--max-num-seqs 8           # Minimal batch size")
        
        # Profile-specific settings
        if profile == 'max_throughput':
            recommendations.append("--enable-chunked-prefill    # Better throughput")
            recommendations.append("--enable-prefix-caching     # Cache common prefixes")
        elif profile == 'low_latency':
            recommendations.append("--disable-log-stats         # Reduce overhead")
            recommendations.append("--enforce-eager             # Disable CUDA graphs for flexibility")
        
        print("\nvLLM Launch Parameters:")
        for rec in recommendations:
            print(f"  {rec}")
        
        # Environment variables
        print("\nRecommended Environment Variables:")
        print("  export CUDA_VISIBLE_DEVICES=0")
        print("  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        if profile == 'max_throughput':
            print("  export VLLM_ATTENTION_BACKEND=FLASH_ATTN")
        
        print("\n" + "=" * 60)
    
    def show_current_status(self, gpu_id: int = 0):
        """Show current GPU status and vLLM readiness"""
        if not self.validate_gpu_id(gpu_id):
            input("\nPress Enter to continue...")
            return

        print("\nCurrent GPU Status:")
        print("=" * 60)

        gpu_info = self.get_gpu_info(gpu_id)
        
        print(f"GPU Model: {gpu_info.get('model', 'Unknown')}")
        print(f"Memory: {gpu_info.get('memory_mb', 0) / 1024:.1f} GB")
        print(f"Current Power Limit: {gpu_info.get('current_power_limit', 0)}W (Max: {gpu_info.get('max_power_limit', 0)}W)")
        print(f"Persistence Mode: {gpu_info.get('persistence', 'Unknown')}")
        print(f"Compute Mode: {gpu_info.get('compute_mode', 'Unknown')}")
        print(f"Current Clocks: {gpu_info.get('graphics_clock', 'N/A')} MHz graphics, {gpu_info.get('memory_clock', 'N/A')} MHz memory")
        
        # Check vLLM readiness
        print("\nvLLM Readiness Check:")
        checks = []
        
        if gpu_info.get('persistence', '').lower() == 'enabled':
            checks.append("✓ Persistence mode enabled (reduces latency)")
        else:
            checks.append("✗ Persistence mode disabled (may increase latency)")
        
        if gpu_info.get('current_power_limit', 0) >= gpu_info.get('max_power_limit', 1) * 0.9:
            checks.append("✓ Power limit near maximum")
        else:
            checks.append("⚠ Power limit not optimized")
        
        compute_mode = gpu_info.get('compute_mode', '')
        if 'Default' in compute_mode or 'Exclusive Process' in compute_mode:
            checks.append(f"✓ Compute mode: {compute_mode}")
        else:
            checks.append(f"⚠ Compute mode may not be optimal: {compute_mode}")
        
        for check in checks:
            print(f"  {check}")
        
        # Last applied profile
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    last_settings = json.load(f)
                    print(f"\nLast Applied Profile: {last_settings.get('profile', 'None')}")
                    print(f"Applied At: {last_settings.get('timestamp', 'Unknown')}")
            except:
                pass
    
    def reset_to_defaults(self, gpu_id: int):
        """Reset GPU to default settings"""
        if not self.validate_gpu_id(gpu_id):
            input("\nPress Enter to continue...")
            return

        print(f"\nResetting GPU {gpu_id} to defaults...")

        results = []
        
        # Reset clocks
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rac']
        success, _ = self.run_command(cmd, check=False)
        if success:
            results.append("✓ Application clocks reset")
        
        # Unlock graphics clocks
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rgc']
        success, _ = self.run_command(cmd, check=False)
        if success:
            results.append("✓ Graphics clocks unlocked")
        
        # Reset power limit to default
        # Query default power limit first
        query_cmd = ['nvidia-smi', '-i', str(gpu_id), '--query-gpu=power.default_limit', '--format=csv,noheader,nounits']
        success, output = self.run_command(query_cmd, check=False)
        if success and output.strip():
            try:
                default_power = int(float(output.strip()))
                cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pl', str(default_power)]
                success, _ = self.run_command(cmd, check=False)
                if success:
                    results.append(f"✓ Power limit reset to default ({default_power}W)")
            except:
                results.append("⚠ Could not reset power limit")
        
        # Set persistence mode off
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pm', '0']
        success, _ = self.run_command(cmd)
        if success:
            results.append("✓ Persistence mode disabled")
        
        # Set compute mode to default
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-c', '0']
        success, _ = self.run_command(cmd)
        if success:
            results.append("✓ Compute mode set to default")
        
        print("\nReset Results:")
        for result in results:
            print(f"  {result}")

def main():
    optimizer = VLLMOptimizer()
    
    # Get GPU count
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    gpu_count = len([l for l in result.stdout.split('\n') if l.startswith('GPU')])
    
    while True:
        os.system('clear')
        print("=" * 60)
        print("vLLM GPU Configuration Optimizer".center(60))
        print("=" * 60)
        print("\nThis tool optimizes NVIDIA GPU settings for vLLM workloads")
        print("\n=== Optimization Profiles ===\n")
        
        profiles_list = list(optimizer.profiles.keys())
        for i, profile_key in enumerate(profiles_list, 1):
            profile = optimizer.profiles[profile_key]
            print(f"{i}. {profile['name']}")
            print(f"   {profile['description']}\n")
        
        print(f"{len(profiles_list) + 1}. Show Current GPU Status")
        print(f"{len(profiles_list) + 2}. Reset GPU to Defaults")
        print("0. Exit")
        
        print("\n" + "=" * 60)
        
        try:
            choice = input("\nEnter choice: ")
            
            if choice == '0':
                break
            
            # Select GPU if multiple available
            gpu_id = 0
            if gpu_count > 1:
                print(f"\nSelect GPU (0-{gpu_count-1}): ", end='')
                try:
                    gpu_id = int(input())
                    if gpu_id < 0 or gpu_id >= gpu_count:
                        gpu_id = 0
                except:
                    gpu_id = 0
            
            if choice.isdigit():
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(profiles_list):
                    # Apply profile
                    profile_key = profiles_list[choice_num - 1]
                    optimizer.apply_profile(gpu_id, profile_key)
                    
                elif choice_num == len(profiles_list) + 1:
                    # Show status
                    optimizer.show_current_status(gpu_id)
                    
                elif choice_num == len(profiles_list) + 2:
                    # Reset
                    if input("\nAre you sure you want to reset GPU to defaults? (y/n): ").lower() == 'y':
                        optimizer.reset_to_defaults(gpu_id)
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            input("\nPress Enter to continue...")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()