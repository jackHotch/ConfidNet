"""Simple example on how to log scalars and images to TensorBoard with TensorFlow 2."""


__author__ = "Michael Gygli"


from io import BytesIO  # Python 3.x
import numpy as np
import tensorflow as tf
import PIL.Image  # Replaces deprecated scipy.misc.toimage




class TensorboardLogger:
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(str(log_dir))


    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""


        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()


    def image_summary(self, tag, images, step):
        """Log a list of images."""
        with self.writer.as_default():
            for i, img in enumerate(images):
                # Convert image to PNG format
                img_pil = PIL.Image.fromarray(np.uint8(img))  # Convert numpy array to PIL Image
                img_buffer = BytesIO()
                img_pil.save(img_buffer, format="PNG")
                img_bytes = img_buffer.getvalue()

                # Log the image
                tf.summary.image(f"{tag}/{i}",
                                 data=tf.image.decode_png(img_bytes, channels=3),
                                 step=step)
            self.writer.flush()


    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor values."""
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step)
            self.writer.flush()
