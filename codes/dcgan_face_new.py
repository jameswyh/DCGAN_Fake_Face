import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

input_path = "dataset/img_align_celeba/"

def load_image(file_name):
    img = PIL.Image.open(file_name)
    img = img.crop([25,65,153,193])
    img = img.resize((64,64))
    data = np.asarray(img, dtype="int32" )
    return data

# print(load_image(input_path + "000001.jpg").shape)
# plt.imshow(load_image(input_path + "000451.jpg"))

buffer_size = 20000
batch_size = 500

# Batch and shuffle the data
train_images = np.array(os.listdir(input_path))
np.random.shuffle(train_images)
train_images = np.split(train_images[:buffer_size],batch_size)

def generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 1024)))
    assert model.output_shape == (None, 4, 4, 1024)

    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5,5), strides = (2,2), padding = "same", use_bias = False, activation = "tanh"))
    assert model.output_shape == (None, 64, 64, 3)

    return model

def discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

generator = generator()
discriminator = discriminator()

# Cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    # print('g_loss:', fake_loss.numpy())
    return fake_loss


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    # print('d_loss:', total_loss.numpy())
    return total_loss

gen_optimizer = tf.keras.optimizers.Adam(2e-4)
disc_optimizer = tf.keras.optimizers.Adam(2e-4)

checkpoint_dir = '.results/face_new_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                 discriminator_optimizer=disc_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 100
noise_dim = 100
example_num = 16

seed = tf.random.normal([example_num, noise_dim])

# `tf.function` causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])


    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))

    images = None

    return gen_loss, disc_loss

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            new_images = []
            for file_name in image_batch:
                new_pic = load_image(input_path + file_name)
                new_images.append(new_pic)

            image_batch = np.array(new_images)
            image_batch = image_batch.reshape(image_batch.shape[0], 64, 64, 3).astype('float32')
            image_batch = (image_batch) / 255 # normalize to [0,1]
            gen_loss, disc_loss = train_step(image_batch)

        # Generate images
        save_images(generator, epoch + 1, seed)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Epoch: {} Time: {} g_loss: {} d_loss: {}'.format(epoch + 1, time.time()-start, gen_loss.numpy(), disc_loss.numpy()))

    # Generate after the final epoch
    save_images(generator, epochs, seed)

def save_images(model, epoch, test_input):
    predictions = model(test_input, training=False).numpy()
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i], interpolation="nearest")
        plt.axis('off')

    plt.savefig('results/face_new_result/image_epoch_{:04d}.png'.format(epoch))
    plt.close()
    #plt.show()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(train_images, EPOCHS)
