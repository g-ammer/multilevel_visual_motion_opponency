package org.janelia.flyem.neuprintloadprocedures.model;

/**
 * A class for counting the number of presynaptic and postsynaptic densities on a neuron.
 */
public class SynapseCounter {

    private long pre;
    private long post;

    /**
     * Class constructor. Counter starts at 0 for both pre and post.
     */
    public SynapseCounter() {
        this.pre = 0;
        this.post = 0;
    }

    /**
     * Class constructor. Counter starts at specified values for
     * pre and post.
     *
     * @param pre  presynaptic density count
     * @param post postsynaptic density count
     */
    public SynapseCounter(long pre, long post) {
        this.pre = pre;
        this.post = post;
    }

    /**
     * @return presynaptic density count
     */
    public long getPre() {
        return this.pre;
    }

    /**
     * @return postsynaptic density count
     */
    public long getPost() {
        return this.post;
    }

    /**
     * Increments the presynaptic density count.
     */
    public void incrementPre() {
        this.pre++;
    }

    /**
     * Increments the postsynaptic density count.
     */
    public void incrementPost() {
        this.post++;
    }

    /**
     * Decrements the presynaptic density count.
     */
    public void decrementPre() {
        if (this.pre > 0) {
            this.pre--;
        }
    }

    /**
     * Decrements the postsynaptic density count.
     */
    public void decrementPost() {
        if (this.post > 0) {
            this.post--;
        }
    }

    @Override
    public String toString() {
        return "{pre: " + this.pre + ", post: " + this.post + "}";
    }

}
