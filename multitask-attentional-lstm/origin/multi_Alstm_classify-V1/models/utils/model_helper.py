import numpy as np



def make_train_feed_dict(model, batch, dropout_keep_prob=0.5):
    # make train feed dict for training
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.seqlen: batch[2],
                 model.keep_pRob: dropout_keep_prob}
    return feed_dict


def run_train_step(model, sess, batch, class_name, dropout_keep_prob):
    feed_dict = make_train_feed_dict(model, batch, dropout_keep_prob)
        
    to_return = [getattr(model,'train_'+class_name+'_op'), model.global_step, getattr(model,'loss_'+class_name), getattr(model,'prediction_'+class_name)]
    
    _, step, loss, predictions = sess.run(to_return, feed_dict)
    
    accuracy = np.sum(np.equal(predictions, batch[1])) / len(predictions)
    return step, loss, accuracy


def run_eval_step(model, sess, class_name, batch,  dropout_keep_prob=1.0):
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.seqlen: batch[2],
                 model.keep_pRob: dropout_keep_prob}
    
    to_return = [getattr(model,'loss_'+class_name), getattr(model,'prediction_'+class_name)]
    loss, predictions = sess.run(to_return, feed_dict)
    return loss, predictions


def get_attn_weight(model, sess, batch, dropout_keep_prob):
    feed_dict = make_train_feed_dict(model, batch, dropout_keep_prob)
    return sess.run(model.alpha, feed_dict)


