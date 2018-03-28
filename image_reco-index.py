#!/usr/bin/python3
import tensorflow as tf
import numpy as np

img_location = "http://dev-collab.tontonserver.com/photo/test.jpg"
img_title = "test.jpg"


def run_inference_on_image(image):
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    node_lookup = NodeLookup()
    top_k = predictions.argsort()[-NUM_TOP_PREDICTIONS:][::-1]
    dict_results = {}

    # create a dictionary
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      dict_results[human_string] = score
    return dict_results

def run_image_recognition(image_path):
    #maybe_download_and_extract()
    return run_inference_on_image(image_path)

def index_new_document(img_title, img_location, result_dictionary):
    result_nested_obj = []
    for key, value in result_dictionary.items():
        result_nested_obj.append({"tag":key, "score":value})

    doc = {
    "title" : img_title,
    "location" : img_location,
    "tags" : result_nested_obj
    }
    res = es.index(index='imagerepository', doc_type='image', body=doc)

# Call the function
if __name__ == '__main__':
    res = index_new_document(
        img_title,
        img_location,
        run_image_recognition(img_location)
    )
    print("res: {}".format(res))

