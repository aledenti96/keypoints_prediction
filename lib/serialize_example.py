import tensorflow as tf

def serialize_example(image,keypoints):

    # i formati particolari come gli array multidimensionali non hanno un formato proprio
    # in cui essere convertiti. Si usa dunque questa funzione per trasformarli in un formato binario 
    # salvato sottoforma di stringa
    # print("Tipo:",type(image))
    serialized_image = tf.io.serialize_tensor(image)
    serialized_coord = tf.io.serialize_tensor(keypoints)

    # salviamo ciò che abbiamo ottenuto sopra in un oggetto chiamato Feature
    bytes_image = tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_image.numpy()]))
    bytes_coord = tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_coord.numpy()]))

    feature = {
        'image': bytes_image,
        'coord': bytes_coord,
    }

    # Ora andiamo a salvare la features in un Example
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    # Prima di ritornare l'example, lo trasformiamo in una stringa
    return example_proto.SerializeToString()