#!/usr/bin/env python3
"""
Integration Validation Example - KISS Implementation

Shows how to use the complete integration validation system.
This example validates that all new systems work together:
- Manifest → Profiles → Analysis → Deviation → Provenance → Results
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def example_quick_health_check():
    """Example: Quick health check of integration systems."""
    print("=== Quick Health Check Example ===")
    
    try:
        from analysis.integration_validation import quick_health_check
        
        # Quick health check - no dependencies required
        status = quick_health_check('config.json')
        
        print(f"System Health Status: {status}")
        
        if status == 'HEALTHY':
            print("✅ All integration systems are healthy and ready!")
        elif status == 'DEGRADED':
            print("⚠️  Some systems are available but degraded")
        else:
            print("❌ Integration systems are unhealthy")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    return True

def example_detailed_health_check():
    """Example: Detailed health check with component breakdown."""
    print("\n=== Detailed Health Check Example ===")
    
    try:
        from analysis.integration_validation import run_integration_health_check
        
        # Detailed health check
        health_report = run_integration_health_check('config.json')
        
        print("Health Report:")
        print(f"  Overall Health: {health_report['overall_health']}")
        print(f"  Configuration Valid: {health_report['configuration_valid']}")
        
        print("\nComponent Availability:")
        for component, available in health_report['components_available'].items():
            status = "✅" if available else "❌"
            print(f"  {status} {component}: {available}")
        
        print("\nImport Status:")
        for module, status in health_report['import_status'].items():
            icon = "✅" if status == 'OK' else "❌"
            print(f"  {icon} {module}: {status}")
            
    except Exception as e:
        print(f"❌ Detailed health check failed: {e}")
        return False
    
    return True

def example_basic_validation():
    """Example: Basic integration validation without reproducibility test."""
    print("\n=== Basic Integration Validation Example ===")
    
    try:
        from analysis.integration_validation import validate_pipeline_integration
        
        # Simple True/False validation
        all_working = validate_pipeline_integration('config.json', output_dir='validation_example')
        
        if all_working:
            print("✅ All pipeline integration systems are working correctly!")
        else:
            print("❌ Some integration systems have issues")
            
        return all_working
        
    except Exception as e:
        print(f"❌ Basic validation failed: {e}")
        return False

def example_full_validation():
    """Example: Full integration validation with all tests."""
    print("\n=== Full Integration Validation Example ===")
    
    try:
        from analysis.integration_validation import run_integration_validation
        
        # Full validation with reproducibility testing
        print("Running comprehensive integration validation...")
        print("This will test:")
        print("  • Manifest system creation and compatibility")
        print("  • Parameter deviation workflow")
        print("  • Provenance tracking")
        print("  • Pipeline integration points")
        print("  • End-to-end reproducibility")
        
        result = run_integration_validation(
            config_path='config.json',
            test_data_dir=None,  # Will create test data
            output_dir='validation_example_full',
            run_reproducibility_test=True
        )
        
        print(f"\n📊 **VALIDATION RESULTS:**")
        print(f"   All Systems Working: {result['all_systems_working']}")
        print(f"   Reproducibility Validated: {result['reproducibility_validated']}")
        print(f"   Execution Time: {result['execution_time_seconds']:.1f}s")
        print(f"   Total Errors: {len(result['errors'])}")
        print(f"   Total Warnings: {len(result['warnings'])}")
        
        print(f"\n🔍 **COMPONENT STATUS:**")
        for component, status_info in result['component_status'].items():
            status = status_info['status']
            icon = "✅" if status == 'PASS' else "⚠️" if status == 'SKIP' else "❌"
            print(f"   {icon} {component}: {status}")
        
        if result['errors']:
            print(f"\n❌ **ERRORS:**")
            for error in result['errors']:
                print(f"   • {error}")
        
        if result['warnings']:
            print(f"\n⚠️  **WARNINGS:**")
            for warning in result['warnings']:
                print(f"   • {warning}")
        
        return result['all_systems_working']
        
    except Exception as e:
        print(f"❌ Full validation failed: {e}")
        return False

def example_workflow_usage():
    """Example: How to use the integration in a real workflow."""
    print("\n=== Integration Workflow Usage Example ===")
    
    # This shows how you would use the integration validation in practice
    print("In a real workflow, you would use it like this:")
    print()
    print("```python")
    print("# 1. Quick health check before starting analysis")
    print("from src.analysis.integration_validation import quick_health_check")
    print("if quick_health_check('config.json') != 'HEALTHY':")
    print("    print('System not ready for analysis')")
    print("    exit(1)")
    print()
    print("# 2. Run your analysis with integrated systems")
    print("from src.analysis.main_pipeline import run_complete_analysis")
    print("result = run_complete_analysis(")
    print("    config_path='config.json',")
    print("    roi_directory='data/roi_files',")
    print("    output_directory='results',")
    print("    create_manifest=True  # Uses manifest system")
    print(")")
    print()
    print("# 3. Validate integration worked correctly")
    print("from src.analysis.integration_validation import validate_pipeline_integration")
    print("integration_ok = validate_pipeline_integration('config.json')")
    print("print(f'Integration validation: {integration_ok}')")
    print("```")
    print()
    print("This ensures all systems (manifests, deviations, provenance) work together!")

def main():
    """Run all integration validation examples."""
    print("🚀 IMC Pipeline Integration Validation Examples")
    print("=" * 60)
    
    examples = [
        example_quick_health_check,
        example_detailed_health_check,
        example_basic_validation,
        example_workflow_usage  # Skip full validation for speed
    ]
    
    results = []
    for example_func in examples:
        try:
            if example_func.__name__ == 'example_workflow_usage':
                example_func()  # This one just prints instructions
                results.append(True)
            else:
                result = example_func()
                results.append(result)
        except Exception as e:
            print(f"❌ Example {example_func.__name__} failed: {e}")
            results.append(False)
    
    print(f"\n📋 **EXAMPLE SUMMARY:**")
    print(f"Examples completed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 **ALL EXAMPLES SUCCESSFUL!**")
        print("\n✨ **INTEGRATION VALIDATION SYSTEM READY FOR USE!**")
        print("\n🔧 **KEY FEATURES:**")
        print("   • Quick health checks for pre-flight validation")
        print("   • Comprehensive system testing with all components")
        print("   • End-to-end reproducibility validation")
        print("   • Detailed error reporting and diagnostics")
        print("   • Simple True/False API for CI/CD integration")
        
        print("\n📁 **FILES CREATED:**")
        print("   • src/analysis/integration_validation.py")
        print("   • validation_example/ (validation outputs)")
        
        print("\n🎯 **CORE FUNCTION:**")
        print("```python")
        print("validation_report = run_integration_validation(")
        print("    config_path='config.json',")
        print("    test_data_dir='test_data/',")
        print("    output_dir='validation_results/'")
        print(")")
        print("# Returns: {'all_systems_working': True, 'component_status': {...}, 'reproducibility_validated': True}")
        print("```")
        
        return True
    else:
        print("\n❌ Some examples failed - check system setup")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)