package org.janelia.flyem.neuprint;

import apoc.convert.Json;
import apoc.create.Create;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.janelia.flyem.neuprint.model.MetaInfo;
import org.janelia.flyem.neuprint.model.Neuron;
import org.janelia.flyem.neuprint.model.Skeleton;
import org.janelia.flyem.neuprint.model.Synapse;
import org.janelia.flyem.neuprint.model.SynapticConnection;
import org.janelia.flyem.neuprintloadprocedures.model.SynapseCounter;
import org.janelia.flyem.neuprintloadprocedures.model.SynapseCounterWithHighPrecisionCounts;
import org.janelia.flyem.neuprintloadprocedures.procedures.LoadingProcedures;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.neo4j.driver.v1.Config;
import org.neo4j.driver.v1.Driver;
import org.neo4j.driver.v1.GraphDatabase;
import org.neo4j.driver.v1.Record;
import org.neo4j.driver.v1.Session;
import org.neo4j.driver.v1.Values;
import org.neo4j.driver.v1.types.Node;
import org.neo4j.driver.v1.types.Point;
import org.neo4j.harness.junit.Neo4jRule;

import java.io.File;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;

import static org.neo4j.driver.v1.Values.parameters;

public class BatchImportTest {

    @ClassRule
    public static Neo4jRule neo4j;
    private static Driver driver;

    static {
        neo4j = new Neo4jRule()
                .withFunction(Json.class)
                .withProcedure(LoadingProcedures.class)
                .withProcedure(Create.class);
    }

    @BeforeClass
    public static void before() {

        final LocalDateTime timeStamp = LocalDateTime.now().truncatedTo(ChronoUnit.SECONDS);

        File swcFile1 = new File("src/test/resources/101.swc");
        File swcFile2 = new File("src/test/resources/102.swc");
        File swcFile3 = new File("src/test/resources/831744.swc");

        File[] arrayOfSwcFiles = new File[]{swcFile1, swcFile2, swcFile3};

        String neuronsJsonPath = "src/test/resources/neuronList.json";

        String synapseJsonPath = "src/test/resources/synapseList.json";

        String connectionsJsonPath = "src/test/resources/connectionsList.json";

        MetaInfo metaInfo = NeuPrintMain.readMetaInfoJson("src/test/resources/testMetaInfo.json");

        driver = GraphDatabase.driver(neo4j.boltURI(), Config.build().withoutEncryption().toConfig());

        Neo4jImporter neo4jImporter = new Neo4jImporter(driver);

        String dataset = "test";

        NeuPrintMain.loadSynapseJsonInBatches(synapseJsonPath,2,neo4jImporter,dataset,false,1.0F,.2D,.8D,true,true,timeStamp);

        NeuPrintMain.loadConnectionJsonInBatches(connectionsJsonPath,2,neo4jImporter,dataset,true,1.0F,.2D,.8D,true,true,timeStamp);

        NeuPrintMain.loadNeuronJsonInBatches(neuronsJsonPath,2,neo4jImporter,dataset,true,1.0F,.2D,.8D,5,true,true,timeStamp);

        NeuPrintMain.loadSkeletonsInBatches(arrayOfSwcFiles, 2, true, neo4jImporter, dataset, 1.0F,.2D,.8D, true, true, timeStamp);

        neo4jImporter.addMetaInfo(dataset, metaInfo, timeStamp);

    }

    @AfterClass
    public static void after() {
        driver.close();
    }

    @Test
    public void synapsesShouldHavePropertiesMatchingInputJson() {

        Session session = driver.session();

        Point preLocationPoint = Values.point(9157, 4287, 2277, 1502).asPoint();
        Node preSynNode = session.readTransaction(tx -> tx.run("MATCH (s:Synapse:PreSyn:`test-Synapse`:`test-PreSyn`:test{location:$location}) RETURN s",
                parameters("location", preLocationPoint))).single().get(0).asNode();

        Assert.assertEquals(1.0, preSynNode.asMap().get("confidence"));
        Assert.assertEquals("pre", preSynNode.asMap().get("type"));
        Assert.assertTrue(preSynNode.asMap().containsKey("roiA"));
        Assert.assertEquals(true, preSynNode.asMap().get("roiA"));

        Point postLocationPoint = Values.point(9157, 4301, 2276, 1535).asPoint();
        Node postSynNode = session.readTransaction(tx -> tx.run("MATCH (s:Synapse:`test-Synapse`:`test-PostSyn`:test:PostSyn{location:$location}) RETURN s",
                parameters("location", postLocationPoint))).single().get(0).asNode();

        Assert.assertEquals(0.1, (double) postSynNode.asMap().get("confidence"), 0.0001);
        Assert.assertEquals("post", postSynNode.asMap().get("type"));
        Assert.assertTrue(postSynNode.asMap().containsKey("roiA"));
    }

    @Test
    public void shouldHaveCorrectNumberOfSynapses() {
        Session session = driver.session();
        int preCount = session.readTransaction(tx -> tx.run("MATCH (s:Synapse:PreSyn:`test-Synapse`:`test-PreSyn`:test) RETURN count(s)")).single().get(0).asInt();
        int postCount = session.readTransaction(tx -> tx.run("MATCH (s:Synapse:PostSyn:`test-Synapse`:`test-PostSyn`:test) RETURN count(s)")).single().get(0).asInt();
        Assert.assertEquals(4, preCount);
        Assert.assertEquals(7, postCount);
    }

    @Test
    public void allSynapsesShouldHaveRequiredProperties() {
        Session session = driver.session();
        int synapsesWithoutRequiredProperties = session.readTransaction(tx -> tx.run("MATCH (s:Synapse) WHERE " +
                "(NOT exists(s.type) OR" +
                " NOT exists(s.confidence) OR" +
                " NOT exists(s.location) OR" +
                " NOT exists(s.timeStamp) OR" +
                " NOT (s:PreSyn OR s:PostSyn) OR" +
                " NOT (s:`test-PreSyn` OR s:`test-PostSyn`) OR" +
                " NOT s:`test-Synapse`)" +
                " RETURN count(s)")).single().get(0).asInt();
        Assert.assertEquals(0, synapsesWithoutRequiredProperties);
    }

    @Test(expected = org.neo4j.driver.v1.exceptions.ClientException.class)
    public void shouldErrorIfAttemptToAddDuplicatePreLocations() {
        Session session = driver.session();
        session.writeTransaction(tx -> tx.run("CREATE (s:Synapse:PreSyn:`test-Synapse`:`test-PreSyn`:test{location:$location}) RETURN s.location", parameters("location", Values.point(9157, 4287, 2277, 1502).asPoint())));
    }

    @Test(expected = org.neo4j.driver.v1.exceptions.ClientException.class)
    public void shouldErrorIfAttemptToAddDuplicatePostLocations() {
        Session session = driver.session();
        session.writeTransaction(tx -> tx.run("CREATE (s:Synapse:PostSyn:`test-Synapse`:`test-PostSyn`:test{location:$location}) RETURN s.location", parameters("location", Values.point(9157, 4301, 2276, 1535).asPoint())));
    }

    @Test
    public void shouldAllowPreAndPostToHaveSameLocation() {
        Session session = driver.session();
        session.writeTransaction(tx -> tx.run("CREATE (s:Synapse:PostSyn:`test-Synapse`:`test-PostSyn`:test{location:$location}) RETURN s.location", parameters("location", Values.point(9157, 4287, 2277, 1502).asPoint())));
        session.writeTransaction(tx -> tx.run("MATCH (s:Synapse:PostSyn:`test-Synapse`:`test-PostSyn`:test{location:$location}) DELETE s", parameters("location", Values.point(9157, 4287, 2277, 1502).asPoint())));
    }

    @Test
    public void synapsesShouldSynapseToCorrectSynapses() {
        Session session = driver.session();

        Point preLocationPoint = Values.point(9157, 4287, 2277, 1502).asPoint();
        int synapsesToCount = session.run("MATCH (s:Synapse:PreSyn:`test-Synapse`:`test-PreSyn`:test{location:$location})-[:SynapsesTo]->(l) RETURN count(l)",
                parameters("location", preLocationPoint)).single().get(0).asInt();
        Assert.assertEquals(2, synapsesToCount);

        int totalSynapsesToCount = session.run("MATCH (s:Synapse)-[st:SynapsesTo]->(t:Synapse) RETURN count(st)").single().get(0).asInt();
        Assert.assertEquals(7, totalSynapsesToCount);

    }

    @Test
    public void allBodiesShouldHaveRequiredProperties() {

        Session session = driver.session();

        int numberOfSegments = session.run("MATCH (n:Segment:test:`test-Segment`) RETURN count(n)").single().get(0).asInt();
        // 10 from neurons json + 2 from skeletons
        Assert.assertEquals(12, numberOfSegments);

        int numberOfSegmentsMissingProperties = session.run("MATCH (n:Segment) WHERE " +
                "(NOT n:test OR" +
                " NOT n:`test-Segment` OR" +
                " NOT exists(n.timeStamp) OR" +
                " NOT exists(n.bodyId) OR" +
                " NOT ((exists(n.somaLocation) AND exists(n.somaRadius)) OR (NOT exists(n.somaLocation) AND NOT exists(n.somaRadius))))" +
                " RETURN count(n)").single().get(0).asInt();
        Assert.assertEquals(0, numberOfSegmentsMissingProperties);
    }

    @Test
    public void shouldNotBeAbleToAddDuplicateSegmentsDueToUniquenessConstraint() {

        Session session = driver.session();

        LocalDateTime timeStamp = LocalDateTime.now().truncatedTo(ChronoUnit.SECONDS);

        Neo4jImporter neo4jImporter = new Neo4jImporter(driver);

        // test uniqueness constraint by trying to add again
        String neuronsJsonPath = "src/test/resources/neuronList.json";
        List<Neuron> neuronList = NeuPrintMain.readNeuronsJson(neuronsJsonPath);
        neo4jImporter.addSegments("test", neuronList, timeStamp);

        int numberOfSegments2 = session.run("MATCH (n:Segment:test:`test-Segment`) RETURN count(n)").single().get(0).asInt();

        // 10 from neurons json + 2 from skeletons
        Assert.assertEquals(12, numberOfSegments2);
    }

    @Test
    public void segmentPropertiesShouldMatchInputJson() {

        Session session = driver.session();

        Node bodyId100569 = session.run("MATCH (n:Segment:`test-Segment`:test{bodyId:100569}) RETURN n").single().get(0).asNode();

        Assert.assertEquals("final", bodyId100569.asMap().get("status"));
        Assert.assertEquals(1031L, bodyId100569.asMap().get("size"));
        Assert.assertEquals("KC-5", bodyId100569.asMap().get("name"));
        Assert.assertEquals("KC", bodyId100569.asMap().get("type"));

        Assert.assertEquals(Values.point(9157, 1.0, 2.0, 3.0).asPoint(), bodyId100569.asMap().get("somaLocation"));
        Assert.assertEquals(5.0, bodyId100569.asMap().get("somaRadius"));

        Assert.assertTrue(bodyId100569.asMap().containsKey("roi1"));
        Assert.assertEquals(true, bodyId100569.asMap().get("roi1"));
        Assert.assertTrue(bodyId100569.asMap().containsKey("roi1"));
        Assert.assertTrue(bodyId100569.asMap().containsKey("roi2"));

        int labelCount = 0;
        Iterable<String> bodyLabels = bodyId100569.labels();

        for (String ignored : bodyLabels) labelCount++;
        Assert.assertEquals(3, labelCount);
    }

    @Test
    public void allSegmentsShouldHaveAStatusWhenListedInJson() {

        Session session = driver.session();

        //bodyId 100541 has no status listed in the json, 2 skeleton bodies have no status, and 2 bodies from smallBodyListWithExtraRois have no status = 5
        int noStatusCount = session.run("MATCH (n:Segment) WHERE NOT exists(n.status) RETURN count(n)").single().get(0).asInt();
        Assert.assertEquals(5, noStatusCount);

    }

    @Test
    public void allSegmentsShouldHaveANameWhenListedInJson() {

        Session session = driver.session();

        int noNameCount = session.run("MATCH (n:Segment) WHERE NOT exists(n.name) RETURN count(n)").single().get(0).asInt();
        Assert.assertEquals(10, noNameCount);

    }

    @Test
    public void allSegmentsWithConnectionsShouldHaveRoiInfoPropertyAndPrePostCounts() {

        Session session = driver.session();

        int roiInfoCount = session.run("MATCH (n:Segment:test:`test-Segment`) WHERE exists(n.roiInfo) RETURN count(n)").single().get(0).asInt();

        Assert.assertEquals(4, roiInfoCount);

        int preCount = session.run("MATCH (n:Segment:test:`test-Segment`) WHERE exists(n.pre) RETURN count(n)").single().get(0).asInt();

        Assert.assertEquals(4, preCount);

        int postCount = session.run("MATCH (n:Segment:test:`test-Segment`) WHERE exists(n.post) RETURN count(n)").single().get(0).asInt();

        Assert.assertEquals(4, postCount);
    }

    @Test
    public void segmentsShouldHaveCorrectPreAndPostCountsAndRoiInfo() {

        Gson gson = new Gson();

        Session session = driver.session();

        Node bodyId8426959 = session.run("MATCH (n:Segment:test:`test-Segment`{bodyId:2589725})<-[r:ConnectsTo]-(s) RETURN s").single().get(0).asNode();

        List<Record> records = session.run("MATCH (n:Segment:test:`test-Segment`{bodyId:8426959})-[:Contains]->(:SynapseSet)-[:Contains]->(s) RETURN s").list();

        for (Record r : records) {
            Node s = (Node) r.asMap().get("s");
            System.out.println(s.asMap());
        }

        Assert.assertEquals(8426959L, bodyId8426959.asMap().get("bodyId"));

        System.out.println(bodyId8426959.asMap());

        Assert.assertEquals(3L, bodyId8426959.asMap().get("post"));
        Assert.assertEquals(2L, bodyId8426959.asMap().get("pre"));
        Map<String, SynapseCounter> synapseCountPerRoi = gson.fromJson((String) bodyId8426959.asMap().get("roiInfo"), new TypeToken<Map<String, SynapseCounter>>() {
        }.getType());

        //should be in lexicographic order
        Set<String> roiSet = new TreeSet<>();
        roiSet.add("roiA");
        roiSet.add("roiB");

        Assert.assertEquals(roiSet, synapseCountPerRoi.keySet());

        Assert.assertEquals(2, synapseCountPerRoi.keySet().size());
        Assert.assertEquals(2, synapseCountPerRoi.get("roiA").getPre());
        Assert.assertEquals(3, synapseCountPerRoi.get("roiA").getPost());
        Assert.assertEquals(1, synapseCountPerRoi.get("roiB").getPost());
        Assert.assertEquals(0, synapseCountPerRoi.get("roiB").getPre());

    }

    @Test
    public void segmentsShouldHaveRoisAsBooleanProperties() {

        Session session = driver.session();

        Node segmentNode = session.run("MATCH (n:Segment:test:`test-Segment`{bodyId:8426959}) RETURN n").single().get(0).asNode();

        Assert.assertTrue(segmentNode.asMap().containsKey("roiA"));
        Assert.assertTrue(segmentNode.asMap().containsKey("roiB"));

    }

    @Test
    public void shouldHaveCorrectConnectsToWeights() {

        Session session = driver.session();

        // weight is equal to number of psds (can have a 1 pre to many post or 1 pre to 1 post connection, but no many pre to 1 post)
        int weight_8426959To2589725 = session.run("MATCH (n:Segment:test:`test-Segment`{bodyId:2589725})<-[r:ConnectsTo]-(s{bodyId:8426959}) RETURN r.weight").single().get(0).asInt();

        Assert.assertEquals(1, weight_8426959To2589725);

        int weight_8426959To26311 = session.run("MATCH (n:Segment:test:`test-Segment`{bodyId:26311})<-[r:ConnectsTo]-(s{bodyId:8426959}) RETURN r.weight").single().get(0).asInt();

        Assert.assertEquals(1, weight_8426959To26311);

        int weight_8426959To831744 = session.run("MATCH (n:Segment:test:`test-Segment`{bodyId:831744})<-[r:ConnectsTo]-(s{bodyId:8426959}) RETURN r.weight").single().get(0).asInt();

        Assert.assertEquals(1, weight_8426959To831744);

        int weight_26311To8426959 = session.run("MATCH (n:Segment:test:`test-Segment`{bodyId:26311})-[r:ConnectsTo]->(s{bodyId:8426959}) RETURN r.weight").single().get(0).asInt();

        Assert.assertEquals(2, weight_26311To8426959);

        int weight_8426959To8426959 = session.run("MATCH (n:Segment:test:`test-Segment`{bodyId:8426959})-[r:ConnectsTo]->(n) RETURN r.weight").single().get(0).asInt();

        Assert.assertEquals(1, weight_8426959To8426959);

    }

    @Test
    public void connectionSetShouldContainAllSynapsesForConnection() {

        Session session = driver.session();

        List<Record> synapseCS_8426959_2589725 = session.run("MATCH (t:ConnectionSet:test:`test-ConnectionSet`{datasetBodyIds:\"test:8426959:2589725\"})-[:Contains]->(s) RETURN s").list();

        for (Record r : synapseCS_8426959_2589725) {
            Node s = (Node) r.asMap().get("s");
            System.out.println(s.asMap());
        }
        Assert.assertEquals(2, synapseCS_8426959_2589725.size());
        Node node1 = (Node) synapseCS_8426959_2589725.get(0).asMap().get("s");
        Node node2 = (Node) synapseCS_8426959_2589725.get(1).asMap().get("s");

        Point synapseLocation1 = (Point) node1.asMap().get("location");
        Point synapseLocation2 = (Point) node2.asMap().get("location");

        Point location1 = Values.point(9157, 4287, 2277, 1542).asPoint();
        Point location2 = Values.point(9157, 4298, 2294, 1542).asPoint();

        Assert.assertTrue(synapseLocation1.equals(location1) || synapseLocation2.equals(location1));
        Assert.assertTrue(synapseLocation1.equals(location2) || synapseLocation2.equals(location2));

        int connectionSetPreCount = session.run("MATCH (n:Neuron:test:`test-Neuron`{bodyId:8426959})<-[:From]-(c:ConnectionSet) RETURN count(c)").single().get(0).asInt();

        Assert.assertEquals(4, connectionSetPreCount);

        int connectionSetPostCount = session.run("MATCH (n:Neuron:test:`test-Neuron`{bodyId:8426959})<-[:To]-(c:ConnectionSet) RETURN count(c)").single().get(0).asInt();

        Assert.assertEquals(2, connectionSetPostCount);

        // weight should be equal to the number of psds per connection (assuming no many pre to one post connections)
        List<Record> connections = session.run("MATCH (n:`test-Neuron`)-[c:ConnectsTo]->(m), (cs:ConnectionSet)-[:Contains]->(s:PostSyn) WHERE cs.datasetBodyIds=\"test:\" + n.bodyId + \":\" + m.bodyId RETURN n.bodyId, m.bodyId, c.weight, cs.datasetBodyIds, count(s)").list();
        for (Record record : connections) {
            Assert.assertEquals(record.asMap().get("c.weight"), record.asMap().get("count(s)"));
        }

        // weightHP should be equal to the number of high-precision psds per connection (assuming no many pre to one post connections)
        List<Record> connectionsHP = session.run("MATCH (n:`test-Neuron`)-[c:ConnectsTo]->(m), (n)<-[:From]-(cs:ConnectionSet)-[:To]->(m), (cs)-[:Contains]->(s:PostSyn) WHERE s.confidence>.81 RETURN n.bodyId, m.bodyId, c.weightHP, count(s)").list();
        for (Record record : connectionsHP) {
            Assert.assertSame(record.asMap().get("c.weightHP"), record.asMap().get("count(s)"));
        }

//        // pre weight should be equal to the number of pre per connection
//        List<Record> connectionsPre = session.run("MATCH (n:`test-Neuron`)-[c:ConnectsTo]->(m), (cs:ConnectionSet)-[:Contains]->(s:PreSyn) WHERE cs.datasetBodyIds=\"test:\" + n.bodyId + \":\" + m.bodyId RETURN n.bodyId, m.bodyId, c.pre, cs.datasetBodyIds, count(s)").list();
//        for (Record record : connectionsPre) {
//            Assert.assertEquals(record.asMap().get("c.pre"), record.asMap().get("count(s)"));
//        }

    }

    @Test
    public void connectionSetsShouldHaveRoiInfoProperty() {
        Session session = driver.session();

        int countOfConnectionSetsWithoutRoiInfo = session.run("MATCH (t:ConnectionSet) WHERE NOT exists(t.roiInfo) RETURN count(t)").single().get("count(t)").asInt();

        Assert.assertEquals(0, countOfConnectionSetsWithoutRoiInfo);

        String roiInfoString = session.readTransaction(tx -> tx.run("MATCH (n:`test-ConnectionSet`{datasetBodyIds:$datasetBodyIds}) RETURN n.roiInfo", parameters("datasetBodyIds", "test:8426959:26311"))).single().get("n.roiInfo").asString();

        Assert.assertNotNull(roiInfoString);

        Gson gson = new Gson();
        Map<String, SynapseCounterWithHighPrecisionCounts> roiInfo = gson.fromJson(roiInfoString, new TypeToken<Map<String, SynapseCounterWithHighPrecisionCounts>>() {
        }.getType());

        Assert.assertEquals(1, roiInfo.size());

        Assert.assertEquals(1, roiInfo.get("roiA").getPre());
        Assert.assertEquals(1, roiInfo.get("roiA").getPreHP());
        Assert.assertEquals(1, roiInfo.get("roiA").getPost());
        Assert.assertEquals(0, roiInfo.get("roiA").getPostHP());

    }

    @Test
    public void shouldHaveCorrectNumberOfSynapseSets() {
        Session session = driver.session();

        List<Record> synapseSets = session.run("MATCH (ss:SynapseSet:`test-SynapseSet`) RETURN ss").list();
        Assert.assertEquals(4, synapseSets.size());

    }

    @Test
    public void shouldHaveCorrectNumberOfConnectionSets() {
        Session session = driver.session();

        List<Record> connectionSets = session.run("MATCH (cs:ConnectionSet:`test-ConnectionSet`) RETURN cs").list();
        Assert.assertEquals(5, connectionSets.size());
    }

    @Test
    public void synapseSetShouldContainAllSynapsesForNeuron() {
        Session session = driver.session();

        List<Record> synapseSS_8426959 = session.run("MATCH (n:`test-Segment`{bodyId:8426959})-[:Contains]->(ss:SynapseSet)-[:Contains]->(s) RETURN s").list();
        Assert.assertEquals(5, synapseSS_8426959.size());

    }

    @Test
    public void shouldHaveCorrectNumberOfSkeletons() {
        Session session = driver.session();

        List<Record> skeletons = session.run("MATCH (s:Skeleton:`test-Skeleton`) RETURN s").list();
        Assert.assertEquals(3, skeletons.size());
    }

    @Test
    public void skeletonsShouldBeConnectedToAppropriateBody() {

        Session session = driver.session();

        Long skeleton101ContainedByBodyId = session.run("MATCH (n:Skeleton:`test-Skeleton`{skeletonId:\"test:101\"})<-[:Contains]-(s) RETURN s.bodyId").single().get(0).asLong();
        Long skeleton102ContainedByBodyId = session.run("MATCH (n:Skeleton:`test-Skeleton`{skeletonId:\"test:102\"})<-[:Contains]-(s) RETURN s.bodyId").single().get(0).asLong();

        Assert.assertEquals(new Long(101), skeleton101ContainedByBodyId);
        Assert.assertEquals(new Long(102), skeleton102ContainedByBodyId);

    }

    @Test
    public void skeletonNodeShouldContainAllSkelNodesForSkeleton() {

        Session session = driver.session();

        Integer skeleton101Degree = session.run("MATCH (n:Skeleton:`test-Skeleton`{skeletonId:\"test:101\"}) WITH n, size((n)-[:Contains]->()) as degree RETURN degree ").single().get(0).asInt();
        Integer skeleton102Degree = session.run("MATCH (n:Skeleton:`test-Skeleton`{skeletonId:\"test:102\"}) WITH n, size((n)-[:Contains]->()) as degree RETURN degree ").single().get(0).asInt();

        Assert.assertEquals(new Integer(50), skeleton101Degree);
        Assert.assertEquals(new Integer(29), skeleton102Degree);
    }

    @Test
    public void skeletonShouldHaveAppropriateNumberOfRootSkelNodes() {

        Session session = driver.session();

        Integer skelNode101NumberOfRoots = session.run("MATCH (n:Skeleton:`test-Skeleton`{skeletonId:\"test:101\"})-[:Contains]->(s:SkelNode:`test-SkelNode`) WHERE NOT (s)<-[:LinksTo]-() RETURN count(s) ").single().get(0).asInt();
        Integer skelNode102RootDegree = session.run("MATCH (n:Skeleton:`test-Skeleton`{skeletonId:\"test:102\"})-[:Contains]->(s:SkelNode:`test-SkelNode`{rowNumber:1}) WITH s, size((s)-[:LinksTo]->()) as degree RETURN degree ").single().get(0).asInt();

        Assert.assertEquals(new Integer(4), skelNode101NumberOfRoots);
        Assert.assertEquals(new Integer(1), skelNode102RootDegree);
    }

    @Test
    public void skelNodesShouldContainAllPropertiesFromSWC() {

        Session session = driver.session();

        Map<String, Object> skelNode101Properties = session.run("MATCH (n:Skeleton:`test-Skeleton`{skeletonId:\"test:101\"})-[:Contains]->(s:SkelNode:`test-SkelNode`{rowNumber:13}) RETURN s.location, s.radius, s.skelNodeId, s.type").list().get(0).asMap();
        Assert.assertEquals(Values.point(9157, 5096, 9281, 1624).asPoint(), skelNode101Properties.get("s.location"));
        Assert.assertEquals(28D, skelNode101Properties.get("s.radius"));
        Assert.assertEquals(0L, skelNode101Properties.get("s.type"));
        Assert.assertEquals("test:101:5096:9281:1624:13", skelNode101Properties.get("s.skelNodeId"));

        Map<String, Object> skelNode831744Properties = session.run("MATCH (n:Skeleton:`test-Skeleton`{skeletonId:\"test:831744\"})-[:Contains]->(s:SkelNode:`test-SkelNode`{rowNumber:315}) RETURN s.location, s.radius, s.skelNodeId, s.type").list().get(0).asMap();
        Assert.assertEquals(Values.point(9157, 12238, 16085, 26505).asPoint(), skelNode831744Properties.get("s.location"));
        Assert.assertEquals(49.141D, (Double) skelNode831744Properties.get("s.radius"), 0.001D);
        Assert.assertEquals(2L, skelNode831744Properties.get("s.type"));
        Assert.assertEquals("test:831744:12238:16085:26505:315", skelNode831744Properties.get("s.skelNodeId"));
    }

    @Test
    public void skelNodesShouldBeProperlyLinked() {

        Session session = driver.session();

        List<Record> skelNode831744Row315LinksTo = session.run("MATCH (n:Skeleton:`test-Skeleton`{skeletonId:\"test:831744\"})-[:Contains]->(s:SkelNode:`test-SkelNode`{rowNumber:315})-[:LinksTo]->(l) RETURN l").list();

        Set<Long> linkedToRowNumbers = skelNode831744Row315LinksTo.stream()
                .map(r -> (Node) r.asMap().get("l"))
                .map(node -> (long) node.asMap().get("rowNumber"))
                .collect(Collectors.toSet());
        Set<Long> expectedRowNumbers = new HashSet<>();
        expectedRowNumbers.add(316L);
        expectedRowNumbers.add(380L);

        Assert.assertEquals(expectedRowNumbers, linkedToRowNumbers);

        List<Record> skelNode831744Row315LinksFrom = session.run("MATCH (n:Skeleton:`test-Skeleton`{skeletonId:\"test:831744\"})-[:Contains]->(s:SkelNode:`test-SkelNode`{rowNumber:315})<-[:LinksTo]-(l) RETURN l").list();

        Set<Long> linkedFromRowNumbers = skelNode831744Row315LinksFrom.stream()
                .map(r -> (Node) r.asMap().get("l"))
                .map(n -> (long) n.asMap().get("rowNumber"))
                .collect(Collectors.toSet());

        Assert.assertTrue(linkedFromRowNumbers.size() == 1 && linkedFromRowNumbers.contains(314L));

    }

    @Test
    public void skelNodeRowNumbersShouldBeDistinctPerSkeleton() {
        Session session = driver.session();
        //check that all row numbers are unique per skeleton
        int rowCount101 = session.readTransaction(tx -> tx.run("MATCH (n:SkelNode) WHERE n.skelNodeId STARTS WITH \"test:101\" WITH DISTINCT n.rowNumber AS rows RETURN count(rows)")).single().get(0).asInt();
        Assert.assertEquals(50, rowCount101);

        int rowCount831744 = session.readTransaction(tx -> tx.run("MATCH (n:SkelNode) WHERE n.skelNodeId STARTS WITH \"test:831744\" WITH DISTINCT n.rowNumber AS rows RETURN count(rows)")).single().get(0).asInt();
        Assert.assertEquals(1679, rowCount831744);
    }

    @Test
    public void metaNodeShouldHaveCorrectSynapseCountsAndSuperLevelRois() {

        Session session = driver.session();

        Node metaNode = session.run("MATCH (n:Meta:test) RETURN n").single().get(0).asNode();
        Assert.assertEquals(4L, metaNode.asMap().get("totalPreCount")); // note that one pre and one post should have been created by connections json
        Assert.assertEquals(7L, metaNode.asMap().get("totalPostCount"));

        List superLevelRois = (List) metaNode.asMap().get("superLevelRois");
        Assert.assertTrue(superLevelRois.contains("roiA") && superLevelRois.contains("roiB"));

        String metaSynapseCountPerRoi = (String) metaNode.asMap().get("roiInfo");
        Gson gson = new Gson();
        Map<String, SynapseCounter> metaSynapseCountPerRoiMap = gson.fromJson(metaSynapseCountPerRoi, new TypeToken<Map<String, SynapseCounter>>() {
        }.getType());

        Assert.assertEquals(5L, metaSynapseCountPerRoiMap.get("roiA").getPost());
        Assert.assertEquals(3L, metaSynapseCountPerRoiMap.get("roiA").getPre());
        Assert.assertEquals(0L, metaSynapseCountPerRoiMap.get("roiB").getPre());
        Assert.assertEquals(3L, metaSynapseCountPerRoiMap.get("roiB").getPost());

        // test to handle ' characters predictably
        Assert.assertEquals(0L, metaSynapseCountPerRoiMap.get("roi'C").getPre());
        Assert.assertEquals(1L, metaSynapseCountPerRoiMap.get("roi'C").getPost());
        // test that all rois are listed in meta
        Assert.assertEquals(3, metaSynapseCountPerRoiMap.keySet().size());
    }

    @Test
    public void dataModelShouldHaveCorrectVersionAndBeConnectedToMetaNode() {

        Session session = driver.session();

        float dataModelVersion = session.run("MATCH (n:Meta)-[:Is]->(d:DataModel) RETURN d.dataModelVersion").single().get(0).asFloat();

        Assert.assertEquals(1.0F, dataModelVersion, .00001);

    }

    @Test
    public void metaNodeShouldHavePreAndPostHPThresholds() {

        Session session = driver.session();

        double preHPThreshold = session.run("MATCH (n:Meta) RETURN n.preHPThreshold").single().get(0).asDouble();
        double postHPThreshold = session.run("MATCH (n:Meta) RETURN n.postHPThreshold").single().get(0).asDouble();

        // test that pre and post HP thresholds are on Meta node
        Assert.assertEquals(.2D, preHPThreshold, .0001);
        Assert.assertEquals(.8D, postHPThreshold, .0001);

    }

    @Test
    public void allNodesShouldHaveDatasetLabelAndTimeStamp() {

        Session session = driver.session();

        int nodeWithoutDatasetLabelCount = session.run("MATCH (n) WHERE NOT n:test RETURN count(n)").single().get(0).asInt();
        // DataModel node does not have dataset label
        Assert.assertEquals(1, nodeWithoutDatasetLabelCount);
        int nodeWithoutTimeStamp = session.run("MATCH (n) WHERE NOT exists(n.timeStamp) RETURN count(n)").single().get(0).asInt();
        // Meta node does not have timeStamp
        Assert.assertEquals(1, nodeWithoutTimeStamp);
    }

    @Test
    public void testThatNeuronLabelsAreAddedToAboveThresholdNeurons() {

        Session session = driver.session();

        int belowThresholdNeuronCount = session.run("MATCH (n:Segment) WHERE (n.pre>=1 OR n.post>=5) AND NOT n:Neuron RETURN count(n)").single().get(0).asInt();

        Assert.assertEquals(0, belowThresholdNeuronCount);

    }

    @Test
    public void shouldAddMetaInfoToMetaNode() {

        Session session = driver.session();

        Node metaNode = session.readTransaction(tx -> tx.run("MATCH (n:Meta{dataset:\"test\"}) RETURN n")).single().get(0).asNode();

        Assert.assertEquals("123456d", metaNode.asMap().get("uuid"));
        Assert.assertEquals("test server", metaNode.asMap().get("dvidServer"));
        Assert.assertEquals("test neuroglancer info", metaNode.asMap().get("neuroglancerInfo"));
        Assert.assertEquals("test host", metaNode.asMap().get("meshHost"));
        Assert.assertEquals("test definitions", metaNode.asMap().get("statusDefinitions"));

    }

    @Test
    public void shouldAddClusterNamesToNeurons() {

        Session session = driver.session();

        int noClusterNameCount = session.readTransaction(tx -> tx.run("MATCH (n:`test-Neuron`) WHERE NOT exists(n.clusterName) RETURN count(n)")).single().get(0).asInt();

        Assert.assertEquals(0, noClusterNameCount);

        String clusterName = session.readTransaction(tx -> tx.run("MATCH (n:`test-Neuron`{bodyId:8426959}) RETURN n.clusterName")).single().get(0).asString();

        Assert.assertEquals("roiA.roiB-roiA", clusterName);

    }
}
