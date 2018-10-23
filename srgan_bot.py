from __future__ import division
from __future__ import print_function

import io
import logging
import os

import numpy as np
import scipy.misc as sic
import tensorflow as tf
from telegram.ext import BaseFilter, Filters, CommandHandler, MessageHandler, Updater

from model import *

# Here should be your token
TOKEN = '<TOKEN>'
checkpoint = './SRGAN_model/model'

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the computation graph
g = tf.Graph()
with g.as_default():
    input_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='input_raw')

    with tf.variable_scope('generator'):
        gen_output = generator(input_raw, 3)

    with tf.name_scope('convert_image'):
        # Deprocess the images outputed from the model [-1, 1] => [0, 1]
        output = deprocess(gen_output)
        # Convert back to uint8
        converted_output = tf.image.convert_image_dtype(output, dtype=tf.uint8, saturate=True)

    # Encode to PNG
    with tf.name_scope('encode_image'):
        output_png = tf.map_fn(tf.image.encode_png, converted_output, dtype=tf.string, name='output_png')

    # Define the weight initiallizer
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    weight_initiallizer = tf.train.Saver(var_list)
    # Define the initialization operation
    init_op = tf.global_variables_initializer()


# Start command
def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id,
                     text='Hi, ' + update.message.from_user.first_name + '  ' + update.message.from_user.last_name + '. ' +
                          'Send images as files to infer photo-realistic natural images ' +
                          'for 4x upscaling factors using deep residual network SRGAN')


# User message handler
def handle_text(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Send an image as a file for its 4x zoom')


# Define an image handler
def handle_image(bot, update):
    message = update.message
    image = message.document
    logger.info("Image %s is received from %s %s", image.file_name, update.message.from_user.first_name,
                update.message.from_user.last_name)
    with io.BytesIO() as fd:
        file_id = bot.get_file(image.file_id)
        file_id.download(out=fd)
        fd.seek(0)
        try:
            im = sic.imread(fd, mode="RGB").astype(np.float32)
        except:
            bot.send_message(chat_id=update.message.chat_id, text='Wrong image')
            logger.info("Wrong image %s from %s %s", image.file_name, update.message.from_user.first_name,
                        update.message.from_user.last_name)
            return
    # Processing images with a resolution of 640*480 takes 10GB of memory(video memory, if you use videocard
    # or system memory, if you use CPU). If you have more, you can increase the resolution
    if (im.shape[0] > 480) or (im.shape[1] > 640):
        bot.send_message(chat_id=update.message.chat_id, text='Image resolution should be no more than 640 * 480')
        logger.info("Exceeding image resolution, file have %s*%s", im.shape[1], im.shape[0])
        return
    im = im / np.max(im)
    im = np.array([im]).astype(np.float32)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=g) as sess:
        # Load the pretrained model
        weight_initiallizer.restore(sess, checkpoint)
        output_png_to_send = sess.run(output_png, feed_dict={input_raw: im})
        name, _ = os.path.splitext(image.file_name)
        filename = name + '_srgan' + '.png'
        with io.BytesIO() as fds:
            fds.name = filename
            fds.write(output_png_to_send[0])
            fds.seek(0)
            bot.send_document(chat_id=update.message.chat_id, document=fds)
    logger.info("Image %s is sent to %s %s", filename, update.message.from_user.first_name,
                update.message.from_user.last_name)


# Log Errors caused by Updates
def error(bot, update, error):
    logger.warning('Update "%s" caused error "%s"', update, error)


# Custom filter for JPG and PNG
class Filter_JPG_PNG(BaseFilter):
    def filter(self, message):
        return message.document.file_name.lower().endswith('.jpg') or message.document.file_name.lower().endswith(
            '.png')


# Create an instance of the class
filter_jpg_png = Filter_JPG_PNG()

# Create the EventHandler and pass it your bot's token
updater = Updater(token=TOKEN)
# Get the dispatcher to register handlers
dispatcher = updater.dispatcher
# Add handlers
text_handler = MessageHandler(Filters.text | Filters.photo, handle_text)
photo_handler = MessageHandler(Filters.document & filter_jpg_png, handle_image)
dispatcher.add_handler(text_handler)
dispatcher.add_handler(photo_handler)
dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(CommandHandler('help', start))
# log all errors
dispatcher.add_error_handler(error)

# The bot is started and runs until we press Ctrl-C on the command line
updater.start_polling()
updater.idle()
