#!/usr/bin/env python3
"""
Inference with a trained ACOC model
====================================
"""

import torch
from torchvision import datasets, transforms
from acoc import ACOCModel


def load_trained_model(checkpoint_path='acoc_mnist.pth'):
    """Load a trained model."""
    print(f"📂 Loading model: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']

    # Create model
    model = ACOCModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  ✓ Model loaded (cycle {checkpoint['cycle']})")
    print(f"  ✓ Parameters: {model.get_total_params():,}")
    print(f"  ✓ Blocks: {len(model.task_blocks)}")

    return model, config


def predict_single(model, image_tensor):
    """Single image prediction."""
    with torch.no_grad():
        image_tensor = image_tensor.to(model.device)
        if image_tensor.dim() == 1:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        output, routing_stats = model(image_tensor)
        probabilities = torch.softmax(output, dim=-1)
        predicted_class = torch.argmax(output, dim=-1).item()
        confidence = probabilities[0, predicted_class].item()

    return predicted_class, confidence, routing_stats


def predict_batch(model, data_loader, num_samples=10):
    """Batch predictions with details."""
    model.eval()

    images, labels = next(iter(data_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    print(f"\n🔍 Predictions on {num_samples} examples:")
    print(f"{'='*70}")

    correct = 0
    for i in range(num_samples):
        image = images[i]
        true_label = labels[i].item()

        pred_class, confidence, routing = predict_single(model, image)
        is_correct = pred_class == true_label
        correct += is_correct

        status = "✓" if is_correct else "✗"
        print(f"{status} Sample {i+1}: True={true_label}, Predicted={pred_class}, "
              f"Confidence={confidence:.2%}")

        # Show routing
        main_block = max(routing, key=routing.get)
        print(f"   → Routed to: {main_block} ({routing[main_block]} samples)")

    print(f"{'='*70}")
    print(f"Accuracy: {correct}/{num_samples} ({100*correct/num_samples:.1f}%)")


def evaluate_full(model, data_loader):
    """Full evaluation on a dataset."""
    model.eval()
    correct = 0
    total = 0

    print(f"\n📊 Full evaluation...")

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(model.device)
            labels = labels.to(model.device)

            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"  ✓ Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def main():
    print("=" * 70)
    print("ACOC Inference")
    print("=" * 70)

    # Load model
    model, config = load_trained_model('acoc_mnist.pth')

    # Load test data (no need for one-hot for inference)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    test_dataset = datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )

    # Example predictions
    predict_batch(model, test_loader, num_samples=10)

    # Full evaluation
    evaluate_full(model, test_loader)

    print(f"\n{'='*70}")
    print("✅ Inference completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
