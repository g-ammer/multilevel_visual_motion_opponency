package org.janelia.flyem.neuprintloadprocedures.model;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * A class for counting the the number of synaptic densities per ROI. Used to produce
 * the roiInfo property on Neuron nodes in neo4j.
 */
public class RoiInfo {

    private Map<String, SynapseCounter> synapseCountsPerRoi = new TreeMap<>();

    /**
     * Class constructor.
     */
    public RoiInfo() {
    }

    public RoiInfo(Map<String, SynapseCounter> roiInfoMap) {
        this.synapseCountsPerRoi.putAll(roiInfoMap);
    }

    /**
     * @return a map of ROIs to {@link SynapseCounter} instances
     */
    Map<String, SynapseCounter> getSynapseCountsPerRoi() {
        return this.synapseCountsPerRoi;
    }

    /**
     * @return the set of all ROIs in this RoiInfo
     */
    public Set<String> getSetOfRois() {
        return this.synapseCountsPerRoi.keySet();
    }

    /**
     * Adds provided ROI to the RoiInfo mapped to a new {@link SynapseCounter} instance
     * initialized with pre and post equal to 0.
     *
     * @param roi ROI name
     */
    private void addSynapseCountsForRoi(String roi) {
        this.synapseCountsPerRoi.put(roi, new SynapseCounter());
    }

    /**
     * Adds provided ROI to the RoiInfo mapped to a new {@link SynapseCounter} instance
     * initialized with pre and post equal to provided values.
     *
     * @param roi  ROI name
     * @param pre  presynaptic density count
     * @param post postsynaptic density count
     */
    public void addSynapseCountsForRoi(String roi, long pre, long post) {
        this.synapseCountsPerRoi.put(roi, new SynapseCounter(pre, post));
    }

    /**
     * @param roi ROI name
     * @return {@link SynapseCounter} for provided ROI
     */
    SynapseCounter getSynapseCountsForRoi(String roi) {
        return this.synapseCountsPerRoi.get(roi);
    }

    public void addRoi(String roi) {
        if (!this.synapseCountsPerRoi.containsKey(roi)) {
            addSynapseCountsForRoi(roi);
        }
    }

    /**
     * Increments the presynaptic density count for the provided ROI by 1.
     *
     * @param roi ROI name
     */
    public void incrementPreForRoi(String roi) {
        addRoi(roi);
        this.synapseCountsPerRoi.get(roi).incrementPre();
    }

    /**
     * Increments the postsynaptic density count for the provided ROI by 1.
     *
     * @param roi ROI name
     */
    public void incrementPostForRoi(String roi) {
        addRoi(roi);
        this.synapseCountsPerRoi.get(roi).incrementPost();
    }

    /**
     * Decrements the presynaptic density count for the provided ROI by 1.
     *
     * @param roi ROI name
     */
    public void decrementPreForRoi(String roi) {
        if (this.synapseCountsPerRoi.containsKey(roi)) {
            SynapseCounter synapseCounter = this.synapseCountsPerRoi.get(roi);
            synapseCounter.decrementPre();

            if (synapseCounter.getPre() + synapseCounter.getPost() == 0) {
                this.synapseCountsPerRoi.remove(roi);
            }

        }
    }

    /**
     * Decrements the postsynaptic density count for the provided ROI by 1.
     *
     * @param roi ROI name
     */
    public void decrementPostForRoi(String roi) {
        if (this.synapseCountsPerRoi.containsKey(roi)) {
            SynapseCounter synapseCounter = this.synapseCountsPerRoi.get(roi);

            synapseCounter.decrementPost();

            if (synapseCounter.getPre() + synapseCounter.getPost() == 0) {
                this.synapseCountsPerRoi.remove(roi);
            }
        }

    }

    /**
     * @return JSON of RoiInfo to be added as an roiInfo property
     * on a node
     */
    public String getAsJsonString() {
        Gson gson = new Gson();
        return gson.toJson(this.synapseCountsPerRoi);
    }

    public SynapseCounter get(String roi) {
        return synapseCountsPerRoi.get(roi);
    }

    @Override
    public String toString() {
        return this.synapseCountsPerRoi.toString();
    }

    public static RoiInfo getRoiInfoFromString(String roiInfo) {
        Gson gson = new Gson();
        Map<String, SynapseCounter> roiInfoMap = gson.fromJson(roiInfo, ROI_INFO_TYPE);
        return new RoiInfo(roiInfoMap);
    }

    public static final Type ROI_INFO_TYPE = new TypeToken<Map<String, SynapseCounter>>() {
    }.getType();
}
