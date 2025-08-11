#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m' 
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Setup DVC environment
setup_dvc_env() {
    print_status $BLUE "🔧 Setting up DVC environment..."
    
    # Create DVC cache directory
    mkdir -p "${DVC_CACHE_DIR:-/tmp/dvc-cache}"
    
    # Configure DVC cache location
    dvc config cache.dir "${DVC_CACHE_DIR:-/tmp/dvc-cache}"
    
    print_status $GREEN "✅ DVC environment configured"
}

# Check if DVC files need updating
check_dvc_changes() {
    print_status $BLUE "📊 Checking for DVC changes..."
    
    if dvc status > /dev/null 2>&1; then
        print_status $GREEN "✅ No DVC changes detected"
        return 0
    else
        print_status $YELLOW "⚠️  DVC changes detected"
        return 1
    fi
}

# Pull DVC data with error handling
pull_dvc_data() {
    print_status $BLUE "📥 Pulling DVC data..."
    
    if dvc pull; then
        print_status $GREEN "✅ DVC data pulled successfully"
        return 0
    else
        print_status $YELLOW "⚠️  DVC pull failed (might be first run)"
        return 1
    fi
}

# Push DVC data with error handling  
push_dvc_data() {
    print_status $BLUE "📤 Pushing DVC data..."
    
    if dvc push; then
        print_status $GREEN "✅ DVC data pushed successfully"
        return 0
    else
        print_status $YELLOW "⚠️  DVC push failed"
        return 1
    fi
}

# Add files to DVC tracking
add_to_dvc() {
    local path=$1
    
    if [ -z "$path" ]; then
        print_status $RED "❌ No path specified for DVC tracking"
        return 1
    fi
    
    print_status $BLUE "📁 Adding $path to DVC..."
    
    if [ -e "$path" ]; then
        if dvc add "$path"; then
            print_status $GREEN "✅ Added $path to DVC tracking"
            return 0
        else
            print_status $RED "❌ Failed to add $path to DVC"
            return 1
        fi
    else
        print_status $YELLOW "⚠️  Path $path does not exist"
        return 1
    fi
}

# Validate DVC setup
validate_dvc() {
    print_status $BLUE "🔍 Validating DVC setup..."
    
    # Check DVC installation
    if ! command -v dvc &> /dev/null; then
        print_status $RED "❌ DVC not installed"
        return 1
    fi
    
    # Check DVC initialization
    if [ ! -d ".dvc" ]; then
        print_status $RED "❌ DVC not initialized"
        return 1
    fi
    
    # Check remote configuration
    if ! dvc remote list | grep -q "local-storage"; then
        print_status $RED "❌ DVC remote not configured"
        return 1
    fi
    
    print_status $GREEN "✅ DVC validation passed"
    return 0
}

# Main function to handle script arguments
main() {
    case "${1:-}" in
        "setup")
            setup_dvc_env
            ;;
        "validate")
            validate_dvc
            ;;
        "pull")
            pull_dvc_data
            ;;
        "push")
            push_dvc_data
            ;;
        "add")
            add_to_dvc "$2"
            ;;
        "check")
            check_dvc_changes
            ;;
        *)
            echo "Usage: $0 {setup|validate|pull|push|add|check}"
            echo "Examples:"
            echo "  $0 setup          - Setup DVC environment"
            echo "  $0 validate       - Validate DVC configuration"
            echo "  $0 pull           - Pull DVC data"
            echo "  $0 push           - Push DVC data"  
            echo "  $0 add data/      - Add directory to DVC tracking"
            echo "  $0 check          - Check for DVC changes"
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi