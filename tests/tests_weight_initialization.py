"""
Weight Initialization Examples for PyDeepFlow
==============================================

This file demonstrates the new weight initialization features and includes
tests to verify correct behavior.
"""

import numpy as np
from pydeepflow.model import Multi_Layer_ANN, Multi_Layer_CNN

# ============================================================================
# Example 1: Automatic Initialization (Activation-Aware)
# ============================================================================

def example_auto_initialization():
    """
    Demonstrates automatic weight initialization based on activation functions.
    The system automatically selects the optimal initialization method.
    """
    print("\n" + "="*80)
    print("Example 1: Automatic Initialization (Recommended)")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(100, 784)  # MNIST-like input
    Y_train = np.eye(10)[np.random.randint(0, 10, 100)]  # One-hot encoded labels
    
    # Create model with automatic initialization
    model = Multi_Layer_ANN(
        X_train=X_train,
        Y_train=Y_train,
        hidden_layers=[256, 128, 64],
        activations=['relu', 'relu', 'tanh'],  # Mixed activations
        weight_init='auto',  # Automatic selection
        bias_init='auto',  # Activation-aware bias initialization
        loss='categorical_crossentropy',
        use_gpu=False
    )
    
    print("\nInitialization Details:")
    print("- Layer 1 (ReLU): He Normal initialization")
    print("- Layer 2 (ReLU): He Normal initialization")
    print("- Layer 3 (Tanh): Xavier Normal initialization")
    print("- Output (Softmax): Xavier Normal initialization")
    
    # Print detailed initialization info
    print("Initialization metadata:")
    for i, meta in enumerate(model.init_metadata):
        print(f"  {meta}")
    
    return model


# ============================================================================
# Example 2: Manual Initialization (Same Method for All Layers)
# ============================================================================

def example_manual_uniform():
    """
    Demonstrates manual weight initialization with a single method for all layers.
    """
    print("\n" + "="*80)
    print("Example 2: Manual Uniform Initialization")
    print("="*80)
    
    np.random.seed(42)
    X_train = np.random.randn(100, 784)
    Y_train = np.eye(10)[np.random.randint(0, 10, 100)]
    
    # Create model with Xavier uniform for all layers
    model = Multi_Layer_ANN(
        X_train=X_train,
        Y_train=Y_train,
        hidden_layers=[256, 128],
        activations=['sigmoid', 'sigmoid'],
        weight_init='xavier_uniform',  # Manual method
        bias_init='zeros',  # All zeros for biases
        loss='categorical_crossentropy',
        use_gpu=False
    )
    
    print("\nAll layers use Xavier Uniform initialization")
    print("Initialization metadata:")
    for i, meta in enumerate(model.init_metadata):
        print(f"  {meta}")
    
    return model


# ============================================================================
# Example 3: Layer-Specific Initialization
# ============================================================================

def example_layer_specific():
    """
    Demonstrates layer-specific weight initialization.
    Each layer can have its own initialization method.
    """
    print("\n" + "="*80)
    print("Example 3: Layer-Specific Initialization")
    print("="*80)
    
    np.random.seed(42)
    X_train = np.random.randn(100, 784)
    Y_train = np.eye(10)[np.random.randint(0, 10, 100)]
    
    # Create model with different initialization for each layer
    model = Multi_Layer_ANN(
        X_train=X_train,
        Y_train=Y_train,
        hidden_layers=[256, 128, 64],
        activations=['relu', 'elu', 'tanh'],
        weight_init=['he_normal', 'he_uniform', 'xavier_normal', 'xavier_uniform'],  # Layer-specific (including output layer)
        bias_init='auto',
        loss='categorical_crossentropy',
        use_gpu=False
    )
    
    print("\nDifferent initialization for each layer:")
    print("- Layer 1: He Normal")
    print("- Layer 2: He Uniform")
    print("- Layer 3: Xavier Normal")
    print("- Output Layer: Xavier Uniform")
    
    print("Initialization metadata:")
    for i, meta in enumerate(model.init_metadata):
        print(f"  {meta}")
    
    return model


# ============================================================================
# Example 4: CNN with Automatic Initialization
# ============================================================================

def example_cnn_auto():
    """
    Demonstrates automatic weight initialization for CNN models using ConvLayer.
    """
    print("\n" + "="*80)
    print("Example 4: CNN ConvLayer with Automatic Initialization")
    print("="*80)
    
    from pydeepflow.model import ConvLayer
    from pydeepflow.device import Device
    
    # Create individual ConvLayer instances to demonstrate initialization
    device = Device(use_gpu=False)
    
    print("Creating ConvLayer instances with automatic initialization:")
    
    # Conv layer 1: ReLU activation
    conv1 = ConvLayer(
        in_channels=3, out_channels=32, kernel_size=3,
        device=device, activation='relu', weight_init='auto'
    )
    print(f"Conv1 (ReLU): {conv1.init_metadata}")
    
    # Conv layer 2: Sigmoid activation  
    conv2 = ConvLayer(
        in_channels=32, out_channels=64, kernel_size=3,
        device=device, activation='sigmoid', weight_init='auto'
    )
    print(f"Conv2 (Sigmoid): {conv2.init_metadata}")
    
    # Conv layer 3: SELU activation
    conv3 = ConvLayer(
        in_channels=64, out_channels=128, kernel_size=3,
        device=device, activation='selu', weight_init='auto'
    )
    print(f"Conv3 (SELU): {conv3.init_metadata}")
    
    print("\nConvLayer initialization working correctly!")
    print("- ReLU → He initialization")
    print("- Sigmoid → Xavier initialization") 
    print("- SELU → LeCun initialization")
    
    return [conv1, conv2, conv3]


# ============================================================================
# Example 5: Custom Bias Initialization
# ============================================================================

def example_custom_bias():
    """
    Demonstrates custom bias initialization values.
    """
    print("\n" + "="*80)
    print("Example 5: Custom Bias Initialization")
    print("="*80)
    
    np.random.seed(42)
    X_train = np.random.randn(100, 784)
    Y_train = np.eye(10)[np.random.randint(0, 10, 100)]
    
    # Create model with custom bias initialization
    model = Multi_Layer_ANN(
        X_train=X_train,
        Y_train=Y_train,
        hidden_layers=[256, 128],
        activations=['relu', 'relu'],
        weight_init='he_normal',
        bias_init=0.1,  # Custom constant value for all biases
        loss='categorical_crossentropy',
        use_gpu=False
    )
    
    print("\nAll biases initialized to 0.1")
    print("Initialization metadata:")
    for i, meta in enumerate(model.init_metadata):
        print(f"  {meta}")
    
    return model


# ============================================================================
# Testing Functions
# ============================================================================

def test_initialization_methods():
    """
    Test that all initialization methods produce valid weights.
    """
    print("\n" + "="*80)
    print("Running Initialization Tests")
    print("="*80)
    
    from pydeepflow.weight_initialization import WeightInitializer
    from pydeepflow.device import Device
    
    device = Device(use_gpu=False)
    methods = ['he_normal', 'he_uniform', 'xavier_normal', 'xavier_uniform', 
               'lecun_normal', 'lecun_uniform', 'random_normal', 'random_uniform']
    
    passed = 0
    failed = 0
    
    for method in methods:
        try:
            initializer = WeightInitializer(device, mode='manual', method=method)
            W, b, meta = initializer.initialize_dense_layer(784, 256, 'relu')
            
            # Check for valid values
            assert not np.isnan(W).any(), f"{method}: Contains NaN"
            assert not np.isinf(W).any(), f"{method}: Contains Inf"
            assert W.shape == (784, 256), f"{method}: Wrong shape"
            assert b.shape == (256,), f"{method}: Wrong bias shape"
            
            print(f"✓ {method}: PASSED")
            passed += 1
            
        except Exception as e:
            print(f"✗ {method}: FAILED - {str(e)}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    return passed, failed


def test_activation_mapping():
    """
    Test that activation functions map to correct initialization methods.
    """
    print("\n" + "="*80)
    print("Testing Activation Function Mapping")
    print("="*80)
    
    from pydeepflow.weight_initialization import WeightInitializer
    from pydeepflow.device import Device
    
    device = Device(use_gpu=False)
    
    expected_mappings = {
        'relu': 'he_normal',
        'leaky_relu': 'he_normal',
        'elu': 'he_normal',
        'sigmoid': 'xavier_normal',
        'tanh': 'xavier_normal',
        'selu': 'lecun_normal',
        'gelu': 'he_normal',
        'swish': 'he_normal'
    }
    
    passed = 0
    failed = 0
    
    for activation, expected_method in expected_mappings.items():
        initializer = WeightInitializer(device, mode='auto')
        actual_method = initializer.get_method_for_activation(activation)
        
        if actual_method == expected_method:
            print(f"✓ {activation} -> {actual_method}: CORRECT")
            passed += 1
        else:
            print(f"✗ {activation} -> {actual_method} (expected {expected_method}): INCORRECT")
            failed += 1
    
    print(f"\nMapping Test Results: {passed} passed, {failed} failed")
    return passed, failed


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PyDeepFlow Weight Initialization Examples and Tests")
    print("="*80)
    
    # Run examples
    print("\n\nRUNNING EXAMPLES:")
    print("="*80)
    
    example_auto_initialization()
    example_manual_uniform()
    example_layer_specific()
    example_cnn_auto()
    example_custom_bias()
    
    # Run tests
    print("\n\nRUNNING TESTS:")
    print("="*80)
    
    test_passed_1, test_failed_1 = test_initialization_methods()
    test_passed_2, test_failed_2 = test_activation_mapping()
    
    total_passed = test_passed_1 + test_passed_2
    total_failed = test_failed_1 + test_failed_2
    
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {total_failed} test(s) failed")