import h5py
from PIL import Image
import numpy as np
import random
import math

def print_h5_structure(name, obj):
    print(name)


categories = [
    b'DRESSES',
    b'FLATS',
    b'HATS',
    b'HEELS',
    b'JACKETS & COATS',
    b'JEANS',
    b'JUMPSUITS',
    b'LINGERIE',
    b'LOAFERS',
    b'PANTS',
    b'SANDALS',
    b'SCARVES',
    b'SHIRTS',
    b'SHORTS',
    b'SKIRTS',
    b'SNEAKERS',
    b'SUITS & BLAZERS',
    b'SWEATERS',
    b'SWIMWEAR',
    b'TOPS',
]

with h5py.File('fashiongen_256_256_train.h5', 'r') as f:
    f.visititems(print_h5_structure)

    

    category = f['input_category']
    image = f['input_image']
    description = f['input_description']
        # Retrieve compression information
    compression = image.compression
    compression_opts = image.compression_opts
    shuffle = image.shuffle
    fletcher32 = image.fletcher32
    chunking = image.chunks

    print(f"Compression Type: {compression}")
    # print(f"Compression Options: {compression_opts}")
    # print(f"Shuffle Enabled: {shuffle}")
    # print(f"Fletcher32 Checksum: {fletcher32}")
    # print(f"Chunking: {chunking}")
    # print(image.dtype)
    indices = np.where(np.isin(category, categories))
    indices_random = np.sort(np.random.choice(indices[0], 100000, replace=False))
    # print(description.shape)



    sample_size = 100000
    batch_size = 200  #

    # Calculate the number of batches
    num_batches = math.ceil(100000 / batch_size)



    with h5py.File('test.h5', 'w') as f_new:
    
        f_new.create_dataset('input_image', shape=(100000, 256, 256, 3), dtype='uint8')
        f_new.create_dataset('input_description', shape=(100000, 1), dtype='S400')  # Assuming descriptions are stored as strings
        
        # Iterate over the sampled indices in batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, sample_size)
            batch_indices = indices_random[start_idx:end_idx]
            
            # Read the batch from the original HDF5 file
            images_batch = image[batch_indices]
            descriptions_batch = description[batch_indices]
            
            # Write the batch to the new HDF5 file
            f_new['input_image'][start_idx:end_idx] = images_batch
            f_new['input_description'][start_idx:end_idx] = descriptions_batch
            
            print(f"Processed batch {i+1}/{num_batches}")
        print("Data filtering and writing completed successfully.")