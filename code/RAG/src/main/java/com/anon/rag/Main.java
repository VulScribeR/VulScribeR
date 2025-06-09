package com.anon.rag;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.anon.rag.model.CodeSnippet;
import com.anon.rag.model.Dataset;
import com.anon.rag.model.LuceneCodeSearchFacade;
import com.anon.rag.model.SearchResult;
import com.anon.rag.utils.DatasetUtils;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

public class Main {

    static List<SearchResult> uniqueDummySearch(Dataset cleans) throws Exception {
        Dataset vulsDataset = DatasetUtils.readFormattedVulsBigVul();
        return uniqueDummySearch(cleans, vulsDataset);
    }

    static List<SearchResult> uniqueDummySearch(Dataset cleans, Dataset vulsDataset) throws Exception {
        ArrayList<CodeSnippet> vuls = new ArrayList<>(vulsDataset.getDataset().values());
        ArrayList<CodeSnippet> cleanItems = new ArrayList<>(cleans.getDataset().values());
        Collections.shuffle(vuls);
        Collections.shuffle(cleanItems);
        ArrayList<SearchResult> searchResults = new ArrayList<>();
        for (int i = 0; i < vuls.size(); i++) {
            CodeSnippet vul = vuls.get(i);
            vul.addMetaData("score", "0");
            searchResults.add(new SearchResult(cleanItems.get(i).getId(), Collections.singletonList(vul)));
        }
        return searchResults;
    }

    static List<SearchResult> dummySearch(Dataset cleans) throws Exception {
        Dataset vulsDataset = DatasetUtils.readFormattedVulsBigVul();
        ArrayList<CodeSnippet> vuls = new ArrayList<>(vulsDataset.getDataset().values());
        ArrayList<CodeSnippet> cleanItems = new ArrayList<>(cleans.getDataset().values());
        Collections.shuffle(vuls);
        Collections.shuffle(cleanItems);
        Random random = new Random(42);
        ArrayList<SearchResult> searchResults = new ArrayList<>();
        HashMap<Integer, Integer> indexCounter = new HashMap<>();
        for (int i = 0; i < cleanItems.size(); i++) { // Has no uniqueness limit, hence each vul can be matched multiple times
            int randomIndex = random.nextInt(vuls.size());
            Integer previousValue = indexCounter.get(randomIndex);
            if (previousValue == null)
                previousValue = 0;
            indexCounter.put(randomIndex, previousValue + 1);
            CodeSnippet vul = vuls.get(randomIndex);
            vul.addMetaData("score", "0");
            searchResults.add(new SearchResult(cleanItems.get(i).getId(), Collections.singletonList(vul)));
        }
        int duplicated = 0;
        int totalDuplicates = 0;
        for (int count : indexCounter.values()){
            if(count > 1){
                System.out.println(count);
                duplicated++;
                totalDuplicates+= count;
            }
        }
        System.out.println(vuls.size());
        System.out.println("Duplicated items:" + duplicated);
        System.out.println("Total Duplicates items:" + totalDuplicates);
        return searchResults;
    }

    // indexes vuls not the fixed version of them
    static LuceneCodeSearchFacade index() throws Exception {
        Dataset fixedVuls = DatasetUtils.readFormattedVulsBigVul();
        LuceneCodeSearchFacade lucene = new LuceneCodeSearchFacade();
        lucene.index(fixedVuls.getDataset().values());
        return lucene;
    }

    static LuceneCodeSearchFacade indexCluster(int clusterNumber) throws Exception {
        String address = "../container_data/bigvul_vuls_cls_" + clusterNumber + ".jsonl";
        Dataset fixedVuls = DatasetUtils.readFormattedVuls(address);
        LuceneCodeSearchFacade lucene = new LuceneCodeSearchFacade();
        lucene.index(fixedVuls.getDataset().values());
        return lucene;
    }

    static LuceneCodeSearchFacade indexPrimeVulCluster(int clusterNumber) throws Exception { // only flaw line ones
        String address = "../container_data/primevul_vuls_cls_" + clusterNumber + "_flaw_only.jsonl";
        Dataset fixedVuls = DatasetUtils.readFormattedVuls(address);
        LuceneCodeSearchFacade lucene = new LuceneCodeSearchFacade();
        lucene.index(fixedVuls.getDataset().values());
        return lucene;
    }

    static LuceneCodeSearchFacade indexFullPrimeVulCluster(int clusterNumber) throws Exception {
        String address = "../container_data/primevul_vuls_cls_" + clusterNumber + ".jsonl";
        Dataset fixedVuls = DatasetUtils.readFormattedVuls(address);
        LuceneCodeSearchFacade lucene = new LuceneCodeSearchFacade();
        lucene.index(fixedVuls.getDataset().values());
        return lucene;
    }

    static List<SearchResult> search(Dataset cleans,
            BiFunction<String, String, SearchResult> searcher,
            Function<CodeSnippet, String> field) {

        return cleans.getDataset()
                .values()
                .stream()
                .map(z -> searcher.apply(z.getId(), field.apply(z)))
                .toList();
    }

    static List<SearchResult> ClusteredSearchWithFunction(List<LuceneCodeSearchFacade> indexPerCluster, Dataset cleans) {
        ArrayList<CodeSnippet> cleansDataset = new ArrayList<>(cleans.getDataset().values());
        double[] scoresSum = {0d, 0d, 0d, 0d, 0d};
        double[] scoresMax = {0d, 0d, 0d, 0d, 0d};
        double[] scoresMin = {0d, 0d, 0d, 0d, 0d};

        List<SearchResult> crossClusterResults = new ArrayList<>();
        for (CodeSnippet searchItem : cleansDataset) {

            List<SearchResult> allClusterResults = new ArrayList<>();
            for (int clusterIndex = 0; clusterIndex < 5; clusterIndex++) {
                String clusterIndexValue = String.valueOf(clusterIndex);
                SearchResult result = indexPerCluster.get(clusterIndex).findSimilarFunction(searchItem.getId(), searchItem.getFunction(), 1);
                result.getScoredCodeSnippets().forEach(z -> z.addMetaData("clusterIndex", clusterIndexValue));
                double score = Double.parseDouble(result.getScoredCodeSnippets().stream().findFirst().map(z -> z.getScore()).orElse("0"));
                scoresSum[clusterIndex] += score;
                scoresMax[clusterIndex] = scoresMax[clusterIndex] < score ? score : scoresMax[clusterIndex];
                scoresMin[clusterIndex] = scoresMin[clusterIndex] > score ? score : scoresMin[clusterIndex];
                allClusterResults.add(result);
            }
            List<CodeSnippet> scoredCodeSnippets = allClusterResults.stream().flatMap(z -> z.getScoredCodeSnippets().stream()).toList();
            crossClusterResults.add(new SearchResult(searchItem.getId(), scoredCodeSnippets));
        }
        System.out.println("PerClusterScoreSum: " + Arrays.toString(scoresSum));
        System.out.println("PerClusterScoreMax: " + Arrays.toString(scoresMax));
        System.out.println("PerClusterScoreMin: " + Arrays.toString(scoresMin));
        return crossClusterResults;
    }

    static List<SearchResult> indexAndSearchWithHeader(Dataset cleans) throws Exception {
        var lucene = index();
        return search(cleans, lucene::findSimilarHeader, CodeSnippet::getHeader);
    }

    static List<SearchResult> indexAndSearchHeader2Code(Dataset cleans) throws Exception {
        var lucene = index();
        return search(cleans, lucene::findHeader2Code, CodeSnippet::getHeader);
    }

    static List<SearchResult> indexAndSearchWithFunction(Dataset cleans) throws Exception {
        var lucene = index();
        return search(cleans, lucene::findSimilarFunction, CodeSnippet::getFunction);
    }

    static void writeToJson(List<SearchResult> results, String fileName) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);

        mapper.writeValue(new File(fileName), results);
    }

    static void generateNaiveDevignFullHeader() throws Exception {
        Dataset cleans = DatasetUtils.readCleanDevign();
        writeToJson(indexAndSearchWithHeader(cleans), "results_full_header2header.json");
    }

    static void generateNaiveBigVulFullHeader() throws Exception {
        Dataset cleans = DatasetUtils.readCleanBigvul();
        writeToJson(indexAndSearchWithHeader(cleans), "results_full_header2header_bigvul.json");
    }

    static void generateNaiveBigVulHeader2Code() throws Exception {
        Dataset cleans = DatasetUtils.readCleanBigvul();
        writeToJson(indexAndSearchWithHeader(cleans), "results_full_header_bigvul.json");
    }

    static void generateNaiveDevignFunction() throws Exception {
        Dataset cleans = DatasetUtils.readCleanDevign();
        writeToJson(indexAndSearchWithFunction(cleans), "results_full_code2code.json");
    }

    static void generateNaiveDevignFunctionClustered() throws Exception {
        Dataset cleans = DatasetUtils.readCleanDevign();
        List<LuceneCodeSearchFacade> indexPerCluster = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            indexPerCluster.add(indexCluster(i));
        }
        writeToJson(ClusteredSearchWithFunction(indexPerCluster, cleans), "results_full_code2code_clustered.json");
    }

    static void generateNaivePrimeVulFunctionClustered() throws Exception {
        Dataset cleans = DatasetUtils.readCleanPrimeVul();
        List<LuceneCodeSearchFacade> indexPerCluster = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            indexPerCluster.add(indexPrimeVulCluster(i));
        }
        writeToJson(ClusteredSearchWithFunction(indexPerCluster, cleans), "results_full_code2code_clustered_primevul_fo.json");
    }

    static void generateNaivePrimeVulFunctionClusteredWithDevignCleans() throws Exception {
        Dataset cleans = DatasetUtils.readCleanDevign();
        List<LuceneCodeSearchFacade> indexPerCluster = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            indexPerCluster.add(indexPrimeVulCluster(i));
        }
        writeToJson(ClusteredSearchWithFunction(indexPerCluster, cleans), "results_full_code2code_clustered_primevul_fo_devign.json");
    }

    static void generateNaivePrimeVulFunctionClusteredFull() throws Exception {
        Dataset cleans = DatasetUtils.readCleanPrimeVul();
        List<LuceneCodeSearchFacade> indexPerCluster = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            indexPerCluster.add(indexFullPrimeVulCluster(i));
        }
        writeToJson(ClusteredSearchWithFunction(indexPerCluster, cleans), "results_full_code2code_clustered_primevul.json");
    }

    static void generateNaiveDevignAndBigVulFunction() throws Exception {
        Dataset cleans = DatasetUtils.readAllClean();
        writeToJson(indexAndSearchWithFunction(cleans), "results_full_code2code_ext.json");
    }

    static void generateNaiveDevignHeader2Code() throws Exception {
        Dataset cleans = DatasetUtils.readCleanDevign();
        writeToJson(indexAndSearchHeader2Code(cleans), "results_full_func.json");
    }

    static void generateUniqueRandom() throws Exception {
        Dataset cleans = DatasetUtils.readCleanDevign();
        writeToJson(uniqueDummySearch(cleans), "results_random.json");
    }

    static void generateUniqueRandomPrimeVul() throws Exception {
        Dataset cleans = DatasetUtils.readCleanPrimeVul();
        Dataset vuls =  DatasetUtils.readFormattedVulsPrimeVul();
        writeToJson(uniqueDummySearch(cleans,  vuls), "results_random_pv.json");
    }

    static void generateRandom() throws Exception {
        Dataset cleans = DatasetUtils.readCleanDevign();
        writeToJson(dummySearch(cleans), "results_random_fair.json");
    }

    /**
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {

        System.out.println("Index N Search");
        // generateUniqueRandom(); # for mutation strategy
        // generateUniqueRandomPrimeVul();
        // generateRandom(); // for random matching with replacement - for w/o retriever settings
        // generateNaiveDevignFunction();   //for matching only based on similaritya - no clustering
        // generateNaiveDevignFunctionClustered();    //for all the clustered strategies 
        // generateNaivePrimeVulFunctionClustered();
        generateNaivePrimeVulFunctionClusteredWithDevignCleans();

        // other experiments
        // generateNaiveDevignFullHeader();
        // generateNaiveDevignAndBigVulFunction();
        // generateNaiveDevignHeader2Code();
        // generateNaiveBigVulHeader2Code();
        // generateNaiveBigVulFullHeader();
    }
    
}
