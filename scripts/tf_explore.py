import tensorflow as tf


# Play around with elementary operations.
a = tf.placeholder('float')
b = tf.placeholder('float')

y = tf.multiply(a,b)
z = tf.multiply(a,1)
out = tf.nn.relu(z)

with tf.Session() as sess:
    print("%f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2})) # eval expressions with parameters for a and b
    print("%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3}))
    # print("Gradient of z w.r.t a is: {:2f}".format( sess.run(tf.gradients(z,a),feed_dict={a:2})))
    print(sess.run(tf.gradients(z,a), feed_dict={a:3}))
    print(sess.run(out, feed_dict={a:-3}))
