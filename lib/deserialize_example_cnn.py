import tensorflow as tf

# funzione per la deserializzazione
# gli esempi vengono passati uno alla volta
def deserialize_example(serialized_example):
    feature = {
        'image': tf.io.FixedLenFeature([],dtype=tf.string),
        'coord': tf.io.FixedLenFeature([],dtype=tf.string),
    }

    # prendiamo l'esempio e lo riportiamo in formato "stringa di bytes"
    bytes_coord = tf.io.parse_single_example(serialized_example,feature)

    # riportiamo quanto abbiamo ottenuto nel formato originale
    img_tensor = tf.io.parse_tensor(bytes_coord['image'],tf.float32)
    coord_tensor = tf.io.parse_tensor(bytes_coord['coord'],tf.float32)

    return img_tensor, coord_tensor