# calibrate.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def find_optimal_threshold():
    """
    Load the trained model, run on the test set, and find the threshold
    that gives the best F1‑score (or you can choose recall if you prefer).
    """
    print("📊 CALIBRATION: Finding optimal decision threshold")
    print("=" * 50)

    # Load model
    try:
        model = load_model('forensic_ai_model.h5')
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return

    # Load test data (no augmentation, just rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(128, 128),
        batch_size=1,          # One by one to get all predictions
        class_mode='binary',
        shuffle=False
    )

    # Collect predictions and true labels
    preds = []
    labels = []
    filenames = []

    for i in range(len(test_generator)):
        x, y = test_generator[i]
        pred = model.predict(x, verbose=0)[0][0]
        preds.append(pred)
        labels.append(y[0])
        filenames.append(test_generator.filenames[i])

    preds = np.array(preds)
    labels = np.array(labels)

    # Evaluate thresholds from 0.00 to 1.00
    thresholds = np.arange(0.00, 1.01, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    best_recall = 0
    best_precision = 0
    results = []

    for t in thresholds:
        y_pred = (preds > t).astype(int)
        tp = np.sum((y_pred == 1) & (labels == 1))
        fp = np.sum((y_pred == 1) & (labels == 0))
        fn = np.sum((y_pred == 0) & (labels == 1))
        tn = np.sum((y_pred == 0) & (labels == 0))

        precision = tp / (tp + fp) if (tp+fp) > 0 else 0
        recall = tp / (tp + fn) if (tp+fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0
        accuracy = (tp + tn) / len(labels)

        results.append((t, precision, recall, f1, accuracy))

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_precision = precision
            best_recall = recall

    # Print summary
    print(f"\n✅ Best threshold (max F1): {best_thresh:.2f}")
    print(f"   F1-score: {best_f1:.3f}")
    print(f"   Precision: {best_precision:.3f}")
    print(f"   Recall: {best_recall:.3f}")

    # Also find threshold that gives recall >= 0.90 if possible
    high_recall_thresh = None
    for t, prec, rec, f1, acc in results:
        if rec >= 0.90 and f1 > 0.5:
            high_recall_thresh = t
            break
    if high_recall_thresh:
        print(f"\n🔍 Threshold for recall >= 90%: {high_recall_thresh:.2f}")

    # Generate classification report at best threshold
    y_pred_best = (preds > best_thresh).astype(int)
    print("\n📋 Classification Report at best threshold:")
    print(classification_report(labels, y_pred_best, target_names=['Authentic', 'Tampered']))

    # Confusion matrix
    cm = confusion_matrix(labels, y_pred_best)
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("              Auth   Tam")
    print(f"Actual Auth   {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"       Tam    {cm[1,0]:3d}   {cm[1,1]:3d}")

    # Plot threshold vs metrics
    plt.figure(figsize=(10, 6))
    thresholds_arr = np.array([r[0] for r in results])
    precisions = np.array([r[1] for r in results])
    recalls = np.array([r[2] for r in results])
    f1s = np.array([r[3] for r in results])
    accs = np.array([r[4] for r in results])

    plt.plot(thresholds_arr, precisions, label='Precision')
    plt.plot(thresholds_arr, recalls, label='Recall')
    plt.plot(thresholds_arr, f1s, label='F1-score')
    plt.plot(thresholds_arr, accs, label='Accuracy')
    plt.axvline(x=best_thresh, color='red', linestyle='--', label=f'Best F1 = {best_thresh:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Calibration')
    plt.legend()
    plt.grid(True)
    plt.savefig('calibration_plot.png')
    print("\n📈 Calibration plot saved as 'calibration_plot.png'")

    # Save the best threshold to a file for use in app.py
    with open('optimal_threshold.txt', 'w') as f:
        f.write(f"{best_thresh:.3f}")
    print(f"💾 Best threshold saved to 'optimal_threshold.txt'")

    return best_thresh


if __name__ == "__main__":
    find_optimal_threshold()