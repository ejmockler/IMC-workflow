#!/usr/bin/env python3
"""
Validation Script for Watershed DNA Segmentation Implementation

Performs syntax and interface validation without requiring external dependencies.
"""

import ast
import sys
from pathlib import Path


def validate_python_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        ast.parse(content)
        return True, "Syntax valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def validate_function_signatures(file_path):
    """Validate function signatures and docstrings."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        functions_found = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions_found.append(node.name)
        
        # Expected functions in watershed_segmentation.py
        expected_functions = [
            'prepare_dna_for_nucleus_detection',
            'detect_nucleus_seeds',
            'perform_watershed_segmentation',
            'aggregate_to_cells',
            'compute_cell_properties',
            'assess_watershed_quality',
            'watershed_pipeline'
        ]
        
        missing_functions = set(expected_functions) - set(functions_found)
        extra_functions = set(functions_found) - set(expected_functions)
        
        return {
            'functions_found': functions_found,
            'expected_functions': expected_functions,
            'missing_functions': list(missing_functions),
            'extra_functions': list(extra_functions),
            'all_expected_present': len(missing_functions) == 0
        }
        
    except Exception as e:
        return {'error': str(e)}


def validate_imports():
    """Validate that imports are properly structured."""
    watershed_file = Path('src/analysis/watershed_segmentation.py')
    multiscale_file = Path('src/analysis/multiscale_analysis.py')
    
    results = {}
    
    # Check watershed file imports
    if watershed_file.exists():
        try:
            with open(watershed_file, 'r') as f:
                content = f.read()
            
            # Check for conditional imports
            has_skimage_check = 'SKIMAGE_AVAILABLE' in content
            has_scipy_check = 'SCIPY_AVAILABLE' in content
            has_conditional_imports = 'try:' in content and 'ImportError:' in content
            
            results['watershed_imports'] = {
                'has_skimage_check': has_skimage_check,
                'has_scipy_check': has_scipy_check,
                'has_conditional_imports': has_conditional_imports,
                'import_safety': all([has_skimage_check, has_scipy_check, has_conditional_imports])
            }
        except Exception as e:
            results['watershed_imports'] = {'error': str(e)}
    
    # Check multiscale integration
    if multiscale_file.exists():
        try:
            with open(multiscale_file, 'r') as f:
                content = f.read()
            
            has_watershed_import = 'watershed_segmentation' in content
            has_watershed_option = "'watershed'" in content
            has_segmentation_method_param = 'segmentation_method' in content
            
            results['multiscale_integration'] = {
                'has_watershed_import': has_watershed_import,
                'has_watershed_option': has_watershed_option,
                'has_segmentation_method_param': has_segmentation_method_param,
                'integration_complete': all([has_watershed_import, has_watershed_option, has_segmentation_method_param])
            }
        except Exception as e:
            results['multiscale_integration'] = {'error': str(e)}
    
    return results


def validate_configuration():
    """Validate configuration has watershed options."""
    config_file = Path('config.json')
    
    if not config_file.exists():
        return {'error': 'config.json not found'}
    
    try:
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check for watershed configuration
        has_watershed_config = 'watershed' in config.get('segmentation', {})
        qc_section = config.get('quality_control', {})
        has_qc_config = 'watershed_segmentation' in qc_section.get('thresholds', {})
        
        watershed_sections = []
        if has_watershed_config:
            watershed_sections = list(config['segmentation']['watershed'].keys())
        
        return {
            'has_watershed_config': has_watershed_config,
            'has_qc_config': has_qc_config,
            'watershed_sections': watershed_sections,
            'config_complete': has_watershed_config and has_qc_config
        }
        
    except Exception as e:
        return {'error': str(e)}


def validate_interface_compatibility():
    """Validate that watershed pipeline has compatible interface with SLIC."""
    watershed_file = Path('src/analysis/watershed_segmentation.py')
    slic_file = Path('src/analysis/slic_segmentation.py')
    
    if not all([watershed_file.exists(), slic_file.exists()]):
        return {'error': 'Missing implementation files'}
    
    try:
        # Parse function signatures
        def get_function_signature(file_path, function_name):
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    args = [arg.arg for arg in node.args.args]
                    return args
            return None
        
        # Compare pipeline signatures
        watershed_args = get_function_signature(watershed_file, 'watershed_pipeline')
        slic_args = get_function_signature(slic_file, 'slic_pipeline')
        
        # Check common parameters
        common_params = [
            'coords', 'ion_counts', 'dna1_intensities', 'dna2_intensities',
            'target_scale_um', 'config', 'cached_cofactors'
        ]
        
        watershed_has_common = all(param in watershed_args for param in common_params) if watershed_args else False
        slic_has_common = all(param in slic_args for param in common_params) if slic_args else False
        
        return {
            'watershed_signature': watershed_args,
            'slic_signature': slic_args,
            'watershed_has_common_params': watershed_has_common,
            'slic_has_common_params': slic_has_common,
            'compatible': watershed_has_common and slic_has_common
        }
        
    except Exception as e:
        return {'error': str(e)}


def main():
    """Run validation checks."""
    print("WATERSHED SEGMENTATION IMPLEMENTATION VALIDATION")
    print("=" * 60)
    
    results = {}
    
    # 1. Syntax validation
    print("\n1. Validating Python syntax...")
    watershed_file = Path('src/analysis/watershed_segmentation.py')
    if watershed_file.exists():
        syntax_valid, syntax_msg = validate_python_syntax(watershed_file)
        results['syntax'] = syntax_valid
        print(f"   {'✓' if syntax_valid else '✗'} {syntax_msg}")
    else:
        print("   ✗ watershed_segmentation.py not found")
        results['syntax'] = False
    
    # 2. Function signature validation
    print("\n2. Validating function signatures...")
    if watershed_file.exists():
        func_validation = validate_function_signatures(watershed_file)
        if 'error' not in func_validation:
            results['functions'] = func_validation['all_expected_present']
            print(f"   {'✓' if func_validation['all_expected_present'] else '✗'} Expected functions present")
            if func_validation['missing_functions']:
                print(f"   Missing: {func_validation['missing_functions']}")
            if func_validation['extra_functions']:
                print(f"   Extra: {func_validation['extra_functions']}")
        else:
            results['functions'] = False
            print(f"   ✗ Error: {func_validation['error']}")
    
    # 3. Import validation
    print("\n3. Validating imports and integration...")
    import_validation = validate_imports()
    
    if 'watershed_imports' in import_validation:
        watershed_imports = import_validation['watershed_imports']
        if 'error' not in watershed_imports:
            results['imports'] = watershed_imports['import_safety']
            print(f"   {'✓' if watershed_imports['import_safety'] else '✗'} Import safety checks")
        else:
            results['imports'] = False
            print(f"   ✗ Import error: {watershed_imports['error']}")
    
    if 'multiscale_integration' in import_validation:
        multiscale_integration = import_validation['multiscale_integration']
        if 'error' not in multiscale_integration:
            results['integration'] = multiscale_integration['integration_complete']
            print(f"   {'✓' if multiscale_integration['integration_complete'] else '✗'} Multiscale integration")
        else:
            results['integration'] = False
            print(f"   ✗ Integration error: {multiscale_integration['error']}")
    
    # 4. Configuration validation
    print("\n4. Validating configuration...")
    config_validation = validate_configuration()
    if 'error' not in config_validation:
        results['config'] = config_validation['config_complete']
        print(f"   {'✓' if config_validation['config_complete'] else '✗'} Configuration complete")
        if config_validation['watershed_sections']:
            print(f"   Watershed sections: {config_validation['watershed_sections']}")
    else:
        results['config'] = False
        print(f"   ✗ Config error: {config_validation['error']}")
    
    # 5. Interface compatibility
    print("\n5. Validating interface compatibility...")
    interface_validation = validate_interface_compatibility()
    if 'error' not in interface_validation:
        results['interface'] = interface_validation['compatible']
        print(f"   {'✓' if interface_validation['compatible'] else '✗'} Interface compatibility")
    else:
        results['interface'] = False
        print(f"   ✗ Interface error: {interface_validation['error']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:.<20} {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Implementation validation successful!")
        print("   Watershed segmentation is properly integrated.")
        return 0
    else:
        print("\n⚠️  Some validation checks failed.")
        print("   Review implementation before testing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())