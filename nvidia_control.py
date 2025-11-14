#!/usr/bin/env python3
"""
NVIDIA GPU Control Panel
A text-based menu system for controlling nvidia-smi settings with persistence
"""

import subprocess
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

class NvidiaGPUController:
    def __init__(self):
        self.settings_file = Path.home() / '.config' / 'nvidiacp' / 'settings.json'
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings = self.load_settings()
        self.gpu_count = self.get_gpu_count()
        
    def load_settings(self) -> Dict[str, Any]:
        """Load persistent settings from file"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def save_settings(self):
        """Save current settings to file"""
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=2)
    
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
        success, output = self.run_command(['nvidia-smi', '-L'])
        if success:
            return len([line for line in output.split('\n') if line.startswith('GPU')])
        return 0

    def get_driver_info(self) -> Tuple[str, str]:
        """Get NVIDIA driver and CUDA versions"""
        driver_version = "Unknown"
        cuda_version = "Unknown"

        success, output = self.run_command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'])
        if success and output.strip():
            driver_version = output.strip().split('\n')[0].strip()

        success, output = self.run_command(['nvidia-smi', '--query-gpu=cuda_version', '--format=csv,noheader'], check=False)
        if success and output.strip():
            cuda_version = output.strip().split('\n')[0].strip()
        else:
            # Fallback: try to get CUDA version from nvidia-smi output
            success, output = self.run_command(['nvidia-smi'], check=False)
            if success:
                for line in output.split('\n'):
                    if 'CUDA Version:' in line:
                        try:
                            cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                        except:
                            pass

        return driver_version, cuda_version

    def validate_gpu_id(self, gpu_id: int) -> bool:
        """Validate that GPU ID exists"""
        if gpu_id < 0 or gpu_id >= self.gpu_count:
            print(f"✗ Invalid GPU ID: {gpu_id}. Valid range is 0-{self.gpu_count - 1}")
            return False
        return True
    
    def get_gpu_info(self) -> str:
        """Get current GPU information"""
        success, output = self.run_command(['nvidia-smi'])
        return output if success else "Failed to get GPU info"
    
    def set_persistence_mode(self, gpu_id: int, enabled: bool):
        """Enable/Disable persistence mode"""
        mode = '1' if enabled else '0'
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pm', mode]
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_persistence'] = enabled
            self.save_settings()
            print(f"✓ Persistence mode {'enabled' if enabled else 'disabled'} for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set persistence mode: {output}")
        input("\nPress Enter to continue...")
    
    def set_power_limit(self, gpu_id: int, watts: int):
        """Set power limit for GPU"""
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pl', str(watts)]
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_power_limit'] = watts
            self.save_settings()
            print(f"✓ Power limit set to {watts}W for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set power limit: {output}")
        input("\nPress Enter to continue...")
    
    def set_gpu_clocks(self, gpu_id: int, mem_clock: Optional[int] = None, graphics_clock: Optional[int] = None):
        """Set GPU memory and graphics clocks"""
        if mem_clock is not None:
            cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-ac', f'{mem_clock},{graphics_clock if graphics_clock else 0}']
            success, output = self.run_command(cmd)
            if success:
                self.settings[f'gpu_{gpu_id}_mem_clock'] = mem_clock
                if graphics_clock:
                    self.settings[f'gpu_{gpu_id}_graphics_clock'] = graphics_clock
                self.save_settings()
                print(f"✓ Clocks set for GPU {gpu_id}")
            else:
                print(f"✗ Failed to set clocks: {output}")
        input("\nPress Enter to continue...")
    
    def reset_gpu_clocks(self, gpu_id: int):
        """Reset GPU clocks to default"""
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rac']
        success, output = self.run_command(cmd)
        if success:
            # Remove clock settings
            self.settings.pop(f'gpu_{gpu_id}_mem_clock', None)
            self.settings.pop(f'gpu_{gpu_id}_graphics_clock', None)
            self.save_settings()
            print(f"✓ GPU {gpu_id} clocks reset to default")
        else:
            print(f"✗ Failed to reset clocks: {output}")
        input("\nPress Enter to continue...")
    
    def set_compute_mode(self, gpu_id: int, mode: int):
        """Set compute mode (0=Default, 1=Exclusive Thread, 2=Prohibited, 3=Exclusive Process)"""
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-c', str(mode)]
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_compute_mode'] = mode
            self.save_settings()
            mode_names = {0: 'Default', 1: 'Exclusive Thread', 2: 'Prohibited', 3: 'Exclusive Process'}
            print(f"✓ Compute mode set to {mode_names.get(mode, 'Unknown')} for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set compute mode: {output}")
        input("\nPress Enter to continue...")
    
    def set_fan_speed(self, gpu_id: int, speed: int):
        """Set fan speed (requires coolbits enabled in X11)"""
        # Note: This requires X11 and coolbits configuration
        cmd = ['nvidia-settings', '-a', f'[gpu:{gpu_id}]/GPUFanControlState=1', '-a', f'[fan:{gpu_id}]/GPUTargetFanSpeed={speed}']
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_fan_speed'] = speed
            self.save_settings()
            print(f"✓ Fan speed set to {speed}% for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set fan speed (may require X11 and coolbits): {output}")
        input("\nPress Enter to continue...")
    
    def enable_ecc(self, gpu_id: int, enabled: bool):
        """Enable/Disable ECC memory (requires reboot)"""
        mode = '1' if enabled else '0'
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-e', mode]
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_ecc'] = enabled
            self.save_settings()
            print(f"✓ ECC {'enabled' if enabled else 'disabled'} for GPU {gpu_id} (requires reboot)")
        else:
            print(f"✗ Failed to set ECC mode: {output}")
        input("\nPress Enter to continue...")
    
    def set_auto_boost(self, gpu_id: int, enabled: bool):
        """Enable/Disable GPU auto boost"""
        mode = '1' if enabled else '0'
        # Try the correct flag first
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '--auto-boost-permission=' + mode]
        success, output = self.run_command(cmd, check=False)
        if success:
            self.settings[f'gpu_{gpu_id}_auto_boost'] = enabled
            self.save_settings()
            print(f"✓ Auto boost {'enabled' if enabled else 'disabled'} for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set auto boost (may not be supported on this GPU): {output}")
        input("\nPress Enter to continue...")
    
    def set_application_clocks(self, gpu_id: int, mem_clock: int, graphics_clock: int):
        """Set application-specific clock speeds"""
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-ac', f'{mem_clock},{graphics_clock}']
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_app_mem_clock'] = mem_clock
            self.settings[f'gpu_{gpu_id}_app_graphics_clock'] = graphics_clock
            self.save_settings()
            print(f"✓ Application clocks set to {mem_clock}MHz mem, {graphics_clock}MHz graphics for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set application clocks: {output}")
        input("\nPress Enter to continue...")
    
    def reset_application_clocks(self, gpu_id: int):
        """Reset application clocks to default"""
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rac']
        success, output = self.run_command(cmd)
        if success:
            self.settings.pop(f'gpu_{gpu_id}_app_mem_clock', None)
            self.settings.pop(f'gpu_{gpu_id}_app_graphics_clock', None)
            self.save_settings()
            print(f"✓ Application clocks reset for GPU {gpu_id}")
        else:
            print(f"✗ Failed to reset application clocks: {output}")
        input("\nPress Enter to continue...")
    
    def set_gpu_reset(self, gpu_id: int):
        """Reset GPU (requires no processes using GPU)"""
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-r']
        success, output = self.run_command(cmd)
        if success:
            print(f"✓ GPU {gpu_id} reset successfully")
        else:
            print(f"✗ Failed to reset GPU: {output}")
        input("\nPress Enter to continue...")
    
    def set_accounting_mode(self, gpu_id: int, enabled: bool):
        """Enable/Disable accounting mode for process tracking"""
        mode = '1' if enabled else '0'
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-am', mode]
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_accounting'] = enabled
            self.save_settings()
            print(f"✓ Accounting mode {'enabled' if enabled else 'disabled'} for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set accounting mode: {output}")
        input("\nPress Enter to continue...")
    
    def clear_accounting_data(self, gpu_id: int):
        """Clear all accounting data"""
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-caa']
        success, output = self.run_command(cmd)
        if success:
            print(f"✓ Accounting data cleared for GPU {gpu_id}")
        else:
            print(f"✗ Failed to clear accounting data: {output}")
        input("\nPress Enter to continue...")
    
    def set_mig_mode(self, gpu_id: int, enabled: bool):
        """Enable/Disable MIG mode (Multi-Instance GPU) - requires reboot"""
        mode = '1' if enabled else '0'
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-mig', mode]
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_mig'] = enabled
            self.save_settings()
            print(f"✓ MIG mode {'enabled' if enabled else 'disabled'} for GPU {gpu_id} (requires reboot)")
        else:
            print(f"✗ Failed to set MIG mode (may not be supported): {output}")
        input("\nPress Enter to continue...")
    
    def query_gpu_details(self, gpu_id: int):
        """Query detailed GPU information"""
        cmd = ['nvidia-smi', '-i', str(gpu_id), '-q']
        success, output = self.run_command(cmd)
        if success:
            print(output)
        else:
            print(f"✗ Failed to query GPU details: {output}")
        input("\nPress Enter to continue...")
    
    def monitor_gpu_live(self):
        """Live monitoring using dmon"""
        print("Starting live GPU monitoring (Press Ctrl+C to stop)...")
        print("Legend: pwr=power, temp=temperature, sm=streaming multiprocessor, mem=memory, enc=encoder, dec=decoder")
        print()
        try:
            subprocess.run(['nvidia-smi', 'dmon', '-s', 'pucmvet'], check=False)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
        input("\nPress Enter to continue...")
    
    def monitor_processes(self):
        """Monitor GPU processes"""
        print("Monitoring GPU processes (Press Ctrl+C to stop)...")
        print()
        try:
            subprocess.run(['nvidia-smi', 'pmon', '-s', 'um'], check=False)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
        input("\nPress Enter to continue...")
    
    def query_supported_clocks(self, gpu_id: int):
        """Query supported clock combinations"""
        cmd = ['nvidia-smi', '-i', str(gpu_id), '-q', '-d', 'SUPPORTED_CLOCKS']
        success, output = self.run_command(cmd)
        if success:
            print(output)
        else:
            print(f"✗ Failed to query supported clocks: {output}")
        input("\nPress Enter to continue...")
    
    def set_gom_mode(self, gpu_id: int, mode: str):
        """Set GPU Operation Mode (all-on, compute, low-dp)"""
        # Try modern flag first
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '--gpu-operation-mode=' + mode]
        success, output = self.run_command(cmd, check=False)
        if not success:
            # Fall back to deprecated flag for older nvidia-smi versions
            cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '--gom=' + mode]
            success, output = self.run_command(cmd, check=False)

        if success:
            self.settings[f'gpu_{gpu_id}_gom'] = mode
            self.save_settings()
            print(f"✓ GOM mode set to {mode} for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set GOM mode (may not be supported on this GPU): {output}")
        input("\nPress Enter to continue...")
    
    def apply_all_settings(self):
        """Apply all saved settings"""
        if not self.settings:
            print("No saved settings to apply")
            time.sleep(2)
            return
            
        print("Applying saved settings...")
        applied_count = 0
        total_count = 0
        
        for key, value in self.settings.items():
            parts = key.split('_')
            if len(parts) >= 3 and parts[0] == 'gpu':
                gpu_id = int(parts[1])
                setting_type = '_'.join(parts[2:])
                total_count += 1
                
                try:
                    if setting_type == 'persistence':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pm', '1' if value else '0']
                        success, _ = self.run_command(cmd)
                        if success: applied_count += 1
                    elif setting_type == 'power_limit':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pl', str(value)]
                        success, _ = self.run_command(cmd)
                        if success: applied_count += 1
                    elif setting_type == 'compute_mode':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-c', str(value)]
                        success, _ = self.run_command(cmd)
                        if success: applied_count += 1
                    elif setting_type == 'ecc':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-e', '1' if value else '0']
                        success, _ = self.run_command(cmd, check=False)
                        if success: applied_count += 1
                    elif setting_type == 'app_mem_clock':
                        graphics_key = f'gpu_{gpu_id}_app_graphics_clock'
                        if graphics_key in self.settings:
                            graphics_clock = self.settings[graphics_key]
                            cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-ac', f'{value},{graphics_clock}']
                            success, _ = self.run_command(cmd, check=False)
                            if success: applied_count += 1
                    elif setting_type == 'accounting':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-am', '1' if value else '0']
                        success, _ = self.run_command(cmd, check=False)
                        if success: applied_count += 1
                    elif setting_type == 'mig':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-mig', '1' if value else '0']
                        success, _ = self.run_command(cmd, check=False)
                        if success: applied_count += 1
                    elif setting_type == 'gom':
                        # Try modern flag first
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '--gpu-operation-mode=' + str(value)]
                        success, _ = self.run_command(cmd, check=False)
                        if not success:
                            # Fall back to deprecated flag
                            cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '--gom=' + str(value)]
                            success, _ = self.run_command(cmd, check=False)
                        if success: applied_count += 1
                except Exception as e:
                    print(f"Failed to apply {key}: {e}")
        
        print(f"✓ Applied {applied_count}/{total_count} settings successfully")
        if applied_count < total_count:
            print("Some settings failed - this is normal for unsupported features")
        time.sleep(2)
    
    def reset_all_gpus(self):
        """Reset all GPUs to default settings"""
        print("Resetting all GPUs to defaults...")
        for gpu_id in range(self.gpu_count):
            # Reset clocks
            subprocess.run(['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rac'], capture_output=True)
            # Reset power limit to default (query default limit first)
            cmd = ['nvidia-smi', '-i', str(gpu_id), '--query-gpu=power.default_limit', '--format=csv,noheader,nounits']
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                try:
                    default_power = int(float(result.stdout.strip()))
                    subprocess.run(['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pl', str(default_power)], capture_output=True)
                except:
                    pass  # If can't parse, skip power reset
            # Set persistence mode off
            subprocess.run(['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pm', '0'], capture_output=True)

        # Clear settings
        self.settings = {}
        self.save_settings()
        print("✓ All GPUs reset to defaults")
        input("\nPress Enter to continue...")
    
    def show_current_settings(self):
        """Display current saved settings"""
        os.system('clear')
        print("=== Current Saved Settings ===\n")
        if not self.settings:
            print("No saved settings")
        else:
            for key, value in sorted(self.settings.items()):
                print(f"{key}: {value}")
        input("\nPress Enter to continue...")
    
    def select_gpu(self) -> Optional[int]:
        """Let user select a GPU"""
        if self.gpu_count == 0:
            print("No NVIDIA GPUs detected!")
            input("\nPress Enter to continue...")
            return None
        
        print("\nSelect GPU:")
        for i in range(self.gpu_count):
            print(f"{i}. GPU {i}")
        print(f"{self.gpu_count}. All GPUs")
        
        try:
            choice = int(input("\nEnter choice: "))
            if 0 <= choice <= self.gpu_count:
                return choice if choice < self.gpu_count else -1  # -1 for all GPUs
        except:
            pass
        
        print("Invalid choice")
        input("\nPress Enter to continue...")
        return None
    
    def display_menu(self):
        """Display main menu"""
        while True:
            os.system('clear')
            driver_ver, cuda_ver = self.get_driver_info()
            print("=" * 60)
            print("NVIDIA GPU Control Panel".center(60))
            print(f"Driver: {driver_ver} | CUDA: {cuda_ver}".center(60))
            print("=" * 60)
            print("\n=== Information & Monitoring ===")
            print("1. Show GPU Information")
            print("2. Query Detailed GPU Info")
            print("3. Live GPU Monitoring (dmon)")
            print("4. Monitor GPU Processes (pmon)")
            print("5. Query Supported Clocks")
            
            print("\n=== Power & Performance ===")
            print("6. Set Persistence Mode")
            print("7. Set Power Limit")
            print("8. Set Application Clocks")
            print("9. Reset Application Clocks")
            print("10. Set GPU Operation Mode (GOM)")
            print("11. Enable/Disable Auto Boost")
            
            print("\n=== Memory & Compute ===")
            print("12. Set Compute Mode")
            print("13. Enable/Disable ECC Memory")
            print("14. Enable/Disable MIG Mode")
            print("15. Enable/Disable Accounting Mode")
            print("16. Clear Accounting Data")
            
            print("\n=== Hardware Control ===")
            print("17. Set Fan Speed")
            print("18. Reset GPU")
            
            print("\n=== System Management ===")
            print("19. Apply All Saved Settings")
            print("20. Reset All GPUs to Defaults")
            print("21. Show Current Settings")
            print("22. Generate Systemd Service")
            
            print("\n=== Specialized Tools ===")
            print("23. vLLM Optimization Tool")
            
            print("\n0. Exit")
            print("\n" + "=" * 60)
            
            try:
                choice = input("\nEnter choice: ")
                
                if choice == '0':
                    break
                
                # Information & Monitoring
                elif choice == '1':
                    os.system('clear')
                    print(self.get_gpu_info())
                    input("\nPress Enter to continue...")
                elif choice == '2':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None and gpu_id != -1:
                        self.query_gpu_details(gpu_id)
                elif choice == '3':
                    self.monitor_gpu_live()
                elif choice == '4':
                    self.monitor_processes()
                elif choice == '5':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None and gpu_id != -1:
                        self.query_supported_clocks(gpu_id)
                
                # Power & Performance
                elif choice == '6':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None:
                        enabled = input("Enable persistence mode? (y/n): ").lower() == 'y'
                        if gpu_id == -1:
                            for i in range(self.gpu_count):
                                self.set_persistence_mode(i, enabled)
                        else:
                            self.set_persistence_mode(gpu_id, enabled)
                elif choice == '7':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None:
                        try:
                            watts = int(input("Enter power limit in watts: "))
                            if gpu_id == -1:
                                for i in range(self.gpu_count):
                                    self.set_power_limit(i, watts)
                            else:
                                self.set_power_limit(gpu_id, watts)
                        except:
                            print("Invalid input")
                            input("\nPress Enter to continue...")
                elif choice == '8':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None:
                        try:
                            mem_clock = int(input("Enter memory clock (MHz): "))
                            graphics_clock = int(input("Enter graphics clock (MHz): "))
                            if gpu_id == -1:
                                for i in range(self.gpu_count):
                                    self.set_application_clocks(i, mem_clock, graphics_clock)
                            else:
                                self.set_application_clocks(gpu_id, mem_clock, graphics_clock)
                        except:
                            print("Invalid input")
                            input("\nPress Enter to continue...")
                elif choice == '9':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None:
                        if gpu_id == -1:
                            for i in range(self.gpu_count):
                                self.reset_application_clocks(i)
                        else:
                            self.reset_application_clocks(gpu_id)
                elif choice == '10':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None:
                        print("\nGPU Operation Modes:")
                        print("1. all-on (All features enabled)")
                        print("2. compute (Compute only, no graphics)")
                        print("3. low-dp (Low double precision)")
                        mode_map = {'1': 'all-on', '2': 'compute', '3': 'low-dp'}
                        mode_choice = input("Enter mode choice: ")
                        if mode_choice in mode_map:
                            if gpu_id == -1:
                                for i in range(self.gpu_count):
                                    self.set_gom_mode(i, mode_map[mode_choice])
                            else:
                                self.set_gom_mode(gpu_id, mode_map[mode_choice])
                        else:
                            print("Invalid choice")
                            input("\nPress Enter to continue...")
                elif choice == '11':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None:
                        enabled = input("Enable auto boost? (y/n): ").lower() == 'y'
                        if gpu_id == -1:
                            for i in range(self.gpu_count):
                                self.set_auto_boost(i, enabled)
                        else:
                            self.set_auto_boost(gpu_id, enabled)
                
                # Memory & Compute
                elif choice == '12':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None:
                        print("\nCompute Modes:")
                        print("0. Default")
                        print("1. Exclusive Thread")
                        print("2. Prohibited")
                        print("3. Exclusive Process")
                        try:
                            mode = int(input("Enter mode: "))
                            if 0 <= mode <= 3:
                                if gpu_id == -1:
                                    for i in range(self.gpu_count):
                                        self.set_compute_mode(i, mode)
                                else:
                                    self.set_compute_mode(gpu_id, mode)
                        except:
                            print("Invalid input")
                            input("\nPress Enter to continue...")
                elif choice == '13':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None:
                        enabled = input("Enable ECC memory? (y/n): ").lower() == 'y'
                        if gpu_id == -1:
                            for i in range(self.gpu_count):
                                self.enable_ecc(i, enabled)
                        else:
                            self.enable_ecc(gpu_id, enabled)
                elif choice == '14':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None:
                        enabled = input("Enable MIG mode? (y/n): ").lower() == 'y'
                        if gpu_id == -1:
                            for i in range(self.gpu_count):
                                self.set_mig_mode(i, enabled)
                        else:
                            self.set_mig_mode(gpu_id, enabled)
                elif choice == '15':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None:
                        enabled = input("Enable accounting mode? (y/n): ").lower() == 'y'
                        if gpu_id == -1:
                            for i in range(self.gpu_count):
                                self.set_accounting_mode(i, enabled)
                        else:
                            self.set_accounting_mode(gpu_id, enabled)
                elif choice == '16':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None:
                        if gpu_id == -1:
                            for i in range(self.gpu_count):
                                self.clear_accounting_data(i)
                        else:
                            self.clear_accounting_data(gpu_id)
                
                # Hardware Control
                elif choice == '17':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None:
                        try:
                            speed = int(input("Enter fan speed (0-100%): "))
                            if 0 <= speed <= 100:
                                if gpu_id == -1:
                                    for i in range(self.gpu_count):
                                        self.set_fan_speed(i, speed)
                                else:
                                    self.set_fan_speed(gpu_id, speed)
                        except:
                            print("Invalid input")
                            input("\nPress Enter to continue...")
                elif choice == '18':
                    gpu_id = self.select_gpu()
                    if gpu_id is not None:
                        if input("Are you sure you want to reset GPU? (y/n): ").lower() == 'y':
                            if gpu_id == -1:
                                for i in range(self.gpu_count):
                                    self.set_gpu_reset(i)
                            else:
                                self.set_gpu_reset(gpu_id)
                
                # System Management
                elif choice == '19':
                    self.apply_all_settings()
                elif choice == '20':
                    if input("Are you sure you want to reset all GPUs? (y/n): ").lower() == 'y':
                        self.reset_all_gpus()
                elif choice == '21':
                    self.show_current_settings()
                elif choice == '22':
                    self.generate_systemd_service()
                elif choice == '23':
                    # Launch vLLM optimizer
                    os.system('clear')
                    print("Launching vLLM Optimization Tool...")
                    time.sleep(1)
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    vllm_script = os.path.join(script_dir, 'vllm_optimizer.py')
                    if os.path.exists(vllm_script):
                        subprocess.run([sys.executable, vllm_script])
                    else:
                        print("vLLM optimizer not found!")
                        input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                input("\nPress Enter to continue...")
    
    def generate_systemd_service(self):
        """Generate systemd service for boot persistence"""
        service_content = f"""[Unit]
Description=NVIDIA GPU Settings Persistence
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 {os.path.abspath(__file__)} --apply-settings
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""
        service_file = Path('/etc/systemd/system/nvidia-settings-persistence.service')
        
        print("\nSystemd service content:")
        print("=" * 60)
        print(service_content)
        print("=" * 60)
        print(f"\nTo install the service, run:")
        print(f"1. sudo tee {service_file} << EOF")
        print(service_content)
        print("EOF")
        print(f"2. sudo systemctl daemon-reload")
        print(f"3. sudo systemctl enable nvidia-settings-persistence.service")
        print(f"4. sudo systemctl start nvidia-settings-persistence.service")
        
        if input("\nAutomatically install service? (requires sudo) (y/n): ").lower() == 'y':
            try:
                # Write service file
                subprocess.run(['sudo', 'tee', str(service_file)], 
                             input=service_content.encode(), 
                             capture_output=True)
                # Reload systemd
                subprocess.run(['sudo', 'systemctl', 'daemon-reload'], capture_output=True)
                # Enable service
                subprocess.run(['sudo', 'systemctl', 'enable', 'nvidia-settings-persistence.service'], 
                             capture_output=True)
                print("✓ Service installed and enabled successfully")
            except Exception as e:
                print(f"✗ Failed to install service: {e}")
        
        input("\nPress Enter to continue...")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--apply-settings':
        # Apply settings mode (for systemd service)
        controller = NvidiaGPUController()
        controller.apply_all_settings()
    else:
        # Interactive menu mode
        controller = NvidiaGPUController()
        if controller.gpu_count == 0:
            print("No NVIDIA GPUs detected!")
            print("Make sure nvidia-smi is installed and working.")
            sys.exit(1)
        
        controller.display_menu()
        print("\nGoodbye!")

if __name__ == "__main__":
    main()