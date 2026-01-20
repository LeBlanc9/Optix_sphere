"""
Demonstration of PhotonBatch merge operations
"""
import optix_sphere._core as osg

def demo_batch_operations():
    print("=== PhotonBatch Merge Operations Demo ===\n")

    # Create some example batches
    batch1 = osg.PhotonBatch(1000)
    batch2 = osg.PhotonBatch(2000)
    batch3 = osg.PhotonBatch(3000)

    print(f"Batch 1 size: {batch1.size()}")
    print(f"Batch 2 size: {batch2.size()}")
    print(f"Batch 3 size: {batch3.size()}")

    # Method 1: append() - Modify batch1 in place (requires GPU copy)
    print("\n--- Method 1: append() ---")
    batch1_copy = osg.PhotonBatch(1000)
    batch1_copy.append(batch2)
    print(f"After batch1.append(batch2): {batch1_copy.size()}")  # 3000
    print(f"Original batch2 unchanged: {batch2.size()}")  # 2000

    # Method 2: swap() - Zero-copy exchange (fastest)
    print("\n--- Method 2: swap() ---")
    empty_batch = osg.PhotonBatch()
    full_batch = osg.PhotonBatch(5000)
    print(f"Before swap - empty: {empty_batch.size()}, full: {full_batch.size()}")

    empty_batch.swap(full_batch)
    print(f"After swap  - empty: {empty_batch.size()}, full: {full_batch.size()}")
    print("  ⚡ Zero-copy operation! Just swapped pointers.")

    # Method 3: PhotonBatch.merge() - Merge multiple batches into one
    print("\n--- Method 3: PhotonBatch.merge() ---")
    merged = osg.PhotonBatch.merge([batch1, batch2, batch3])
    print(f"Merged batch size: {merged.size()}")  # 6000
    print(f"Original batches unchanged:")
    print(f"  batch1: {batch1.size()}")
    print(f"  batch2: {batch2.size()}")
    print(f"  batch3: {batch3.size()}")

    # Practical use case: Merge simulation results
    print("\n--- Practical Example: Merge Simulation Results ---")
    print("Scenario: Combine specular + negative + positive boundary batches")

    # Simulate result batches
    specular = osg.PhotonBatch(500)      # Specular reflections
    negative = osg.PhotonBatch(1500)     # Exit from -Z (reflected)
    positive = osg.PhotonBatch(3000)     # Exit from +Z (transmitted)

    # Merge all output photons into one batch
    all_photons = osg.PhotonBatch.merge([specular, negative, positive])
    print(f"Total output photons: {all_photons.size()}")

    # Or just merge reflected photons
    all_reflected = osg.PhotonBatch.merge([specular, negative])
    print(f"Total reflected photons: {all_reflected.size()}")

    # Performance note
    print("\n--- Performance Tips ---")
    print("1. swap()   - Fastest (zero-copy), use when you want to transfer ownership")
    print("2. append() - Moderate (GPU copy), use to accumulate results")
    print("3. merge()  - Flexible (GPU copy), use to combine multiple batches")
    print("\n✓ All operations work on GPU memory - no CPU transfer needed!")


if __name__ == "__main__":
    demo_batch_operations()
