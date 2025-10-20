#!/usr/bin/env python3
"""
Validation script for single-stain protocols module structure and integration.

This script validates:
1. Module imports and dependencies
2. Class and function definitions
3. Integration points with existing modules
4. API compatibility and structure
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set

def validate_module_structure():
    """Validate the structure of the single_stain_protocols module."""
    print("🔍 Validating single_stain_protocols.py module structure...")
    
    module_path = Path("src/analysis/single_stain_protocols.py")
    
    if not module_path.exists():
        print(f"❌ Module not found: {module_path}")
        return False
    
    # Parse the module AST
    with open(module_path, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"❌ Syntax error in module: {e}")
        return False
    
    # Extract module components
    imports = []
    classes = []
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
    
    print(f"✅ Module parsed successfully")
    print(f"   📦 Imports: {len(imports)}")
    print(f"   🏗️  Classes: {len(classes)}")
    print(f"   🔧 Functions: {len(functions)}")
    
    # Validate expected components
    expected_classes = [
        'SingleStainProtocol',
        'SpilloverEstimationResult',
        'SingleStainProtocolError'
    ]
    
    expected_functions = [
        'create_standard_protocols',
        'validate_single_stain_data', 
        'estimate_spillover_from_single_stains',
        'create_spillover_correction_pipeline',
        'save_single_stain_analysis',
        'load_single_stain_analysis'
    ]
    
    missing_classes = [cls for cls in expected_classes if cls not in classes]
    missing_functions = [func for func in expected_functions if func not in functions]
    
    if missing_classes:
        print(f"❌ Missing classes: {missing_classes}")
        return False
    
    if missing_functions:
        print(f"❌ Missing functions: {missing_functions}")
        return False
    
    print("✅ All expected classes and functions found")
    return True


def validate_integration_imports():
    """Validate imports from existing pipeline modules."""
    print("\n🔗 Validating integration imports...")
    
    module_path = Path("src/analysis/single_stain_protocols.py")
    
    with open(module_path, 'r') as f:
        content = f.read()
    
    # Check for expected integration imports
    expected_imports = [
        'spillover_correction',
        'uncertainty_propagation', 
        'artifact_detection',
        'ion_count_processing',
        'quality_control'
    ]
    
    missing_imports = []
    for expected in expected_imports:
        if f"from .{expected} import" not in content:
            missing_imports.append(expected)
    
    if missing_imports:
        print(f"❌ Missing integration imports: {missing_imports}")
        return False
    
    print("✅ All integration imports present")
    
    # Check for specific imported components
    required_components = [
        'SpilloverMatrix',
        'SpilloverCorrectionError', 
        'UncertaintyMap',
        'UncertaintyConfig',
        'DetectorConfig'
    ]
    
    missing_components = []
    for component in required_components:
        if component not in content:
            missing_components.append(component)
    
    if missing_components:
        print(f"❌ Missing imported components: {missing_components}")
        return False
    
    print("✅ All required components imported")
    return True


def validate_existing_modules():
    """Validate that existing modules have expected functions."""
    print("\n📋 Validating existing module interfaces...")
    
    module_checks = {
        "src/analysis/spillover_correction.py": [
            "SpilloverMatrix",
            "estimate_spillover_matrix", 
            "correct_spillover",
            "validate_spillover_correction"
        ],
        "src/analysis/uncertainty_propagation.py": [
            "UncertaintyMap",
            "UncertaintyConfig",
            "create_base_uncertainty",
            "propagate_through_spillover_correction"
        ],
        "src/analysis/artifact_detection.py": [
            "DetectorConfig",
            "detect_and_correct_artifacts",
            "correct_detector_nonlinearity"
        ],
        "src/analysis/quality_control.py": [
            "monitor_calibration_channels",
            "check_background_levels",
            "detect_spatial_artifacts"
        ]
    }
    
    all_valid = True
    
    for module_path, expected_items in module_checks.items():
        path = Path(module_path)
        
        if not path.exists():
            print(f"❌ Module not found: {module_path}")
            all_valid = False
            continue
        
        with open(path, 'r') as f:
            content = f.read()
        
        missing_items = []
        for item in expected_items:
            if f"class {item}" not in content and f"def {item}" not in content:
                missing_items.append(item)
        
        if missing_items:
            print(f"❌ {module_path} missing: {missing_items}")
            all_valid = False
        else:
            print(f"✅ {module_path} interface complete")
    
    return all_valid


def validate_api_compatibility():
    """Validate API compatibility and method signatures."""
    print("\n🔌 Validating API compatibility...")
    
    module_path = Path("src/analysis/single_stain_protocols.py")
    
    with open(module_path, 'r') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    # Extract function signatures
    function_signatures = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            function_signatures[node.name] = args
    
    # Validate key function signatures
    expected_signatures = {
        'create_standard_protocols': [],  # No required args
        'estimate_spillover_from_single_stains': ['single_stain_measurements'],
        'create_spillover_correction_pipeline': ['single_stain_result', 'ion_count_data'],
        'save_single_stain_analysis': ['result', 'output_path'],
        'load_single_stain_analysis': ['file_path']
    }
    
    signature_issues = []
    
    for func_name, expected_args in expected_signatures.items():
        if func_name in function_signatures:
            actual_args = function_signatures[func_name]
            for expected_arg in expected_args:
                if expected_arg not in actual_args:
                    signature_issues.append(f"{func_name} missing arg: {expected_arg}")
        else:
            signature_issues.append(f"Function not found: {func_name}")
    
    if signature_issues:
        print(f"❌ API signature issues: {signature_issues}")
        return False
    
    print("✅ API signatures compatible")
    return True


def validate_dataclass_definitions():
    """Validate dataclass definitions for proper structure."""
    print("\n📊 Validating dataclass definitions...")
    
    module_path = Path("src/analysis/single_stain_protocols.py")
    
    with open(module_path, 'r') as f:
        content = f.read()
    
    # Check for dataclass decorators
    required_dataclasses = [
        'SingleStainProtocol',
        'SpilloverEstimationResult'
    ]
    
    dataclass_issues = []
    
    for dataclass_name in required_dataclasses:
        # Look for @dataclass decorator before class definition
        class_pattern = f"class {dataclass_name}"
        class_pos = content.find(class_pattern)
        
        if class_pos == -1:
            dataclass_issues.append(f"Class not found: {dataclass_name}")
            continue
        
        # Check for @dataclass decorator in the preceding lines
        preceding_content = content[:class_pos]
        lines = preceding_content.split('\n')
        
        # Look in last few lines before class definition
        has_dataclass_decorator = False
        for line in lines[-5:]:
            if "@dataclass" in line:
                has_dataclass_decorator = True
                break
        
        if not has_dataclass_decorator:
            dataclass_issues.append(f"{dataclass_name} missing @dataclass decorator")
    
    if dataclass_issues:
        print(f"❌ Dataclass issues: {dataclass_issues}")
        return False
    
    print("✅ Dataclass definitions valid")
    return True


def validate_error_handling():
    """Validate error handling and exception definitions."""
    print("\n⚠️  Validating error handling...")
    
    module_path = Path("src/analysis/single_stain_protocols.py")
    
    with open(module_path, 'r') as f:
        content = f.read()
    
    # Check for custom exception
    if "class SingleStainProtocolError(Exception):" not in content:
        print("❌ Custom exception SingleStainProtocolError not defined")
        return False
    
    # Check for proper error handling patterns
    error_patterns = [
        "raise SingleStainProtocolError",
        "try:",
        "except"
    ]
    
    missing_patterns = []
    for pattern in error_patterns:
        if pattern not in content:
            missing_patterns.append(pattern)
    
    if missing_patterns:
        print(f"❌ Missing error handling patterns: {missing_patterns}")
        return False
    
    print("✅ Error handling patterns present")
    return True


def validate_logging_integration():
    """Validate logging integration."""
    print("\n📝 Validating logging integration...")
    
    module_path = Path("src/analysis/single_stain_protocols.py")
    
    with open(module_path, 'r') as f:
        content = f.read()
    
    # Check for logging setup
    if "import logging" not in content:
        print("❌ Logging module not imported")
        return False
    
    if "logger = logging.getLogger(__name__)" not in content:
        print("❌ Logger not configured")
        return False
    
    # Check for logging calls
    logging_patterns = ["logger.info", "logger.debug", "logger.warning", "logger.error"]
    found_patterns = [pattern for pattern in logging_patterns if pattern in content]
    
    if len(found_patterns) < 3:
        print(f"❌ Insufficient logging calls found: {found_patterns}")
        return False
    
    print(f"✅ Logging integration complete ({len(found_patterns)} call types)")
    return True


def run_comprehensive_validation():
    """Run comprehensive validation of the single-stain protocols module."""
    print("🧪 Starting comprehensive single-stain protocols validation")
    print("=" * 70)
    
    validations = [
        ("Module Structure", validate_module_structure),
        ("Integration Imports", validate_integration_imports),
        ("Existing Modules", validate_existing_modules),
        ("API Compatibility", validate_api_compatibility),
        ("Dataclass Definitions", validate_dataclass_definitions),
        ("Error Handling", validate_error_handling),
        ("Logging Integration", validate_logging_integration)
    ]
    
    results = {}
    all_passed = True
    
    for name, validation_func in validations:
        try:
            result = validation_func()
            results[name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {name} validation failed with error: {e}")
            results[name] = False
            all_passed = False
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {name}")
    
    print("=" * 70)
    
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ single_stain_protocols.py is ready for integration")
        return True
    else:
        failed_count = sum(1 for result in results.values() if not result)
        print(f"❌ {failed_count}/{len(validations)} validations failed")
        print("⚠️  Module needs fixes before integration")
        return False


if __name__ == "__main__":
    success = run_comprehensive_validation()
    
    if success:
        print(f"\n✅ Single-stain protocols module validation PASSED")
        sys.exit(0)
    else:
        print(f"\n❌ Single-stain protocols module validation FAILED")
        sys.exit(1)