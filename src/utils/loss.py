import jax, jax.numpy as jnp

def attention_rank_loss(scores, labels, *, where=None):
    """ Attention rank loss, modeled to mimic the rax codebase. """

    # applies mask so that masked elements do not count towards the loss.
    if where is not None:
        labels = jnp.where(where, labels, -jnp.ones_like(labels) * jnp.inf)
        scores = jnp.where(where, scores, -jnp.ones_like(scores) * jnp.inf)

    scores_log_sofmax = jax.nn.log_softmax(scores, axis=-1)
    labels_softmax = jax.nn.softmax(labels, axis=-1)

    loss = -jnp.sum(scores_log_sofmax * labels_softmax, axis=-1, where=where)
    return jnp.mean(loss)