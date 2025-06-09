package com.anon.rag.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.DirectoryIteratorException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import com.anon.rag.model.Dataset;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.opencsv.CSVReaderHeaderAware;

public class DatasetUtils {

    private DatasetUtils() {
    }

    public static Dataset readCleanDevign() {
        return readCleanDevign("../container_data/train_devign/");
    }

    public static List<String> listFiles(String directoryPath) {
        Path dir = Paths.get(directoryPath).toAbsolutePath();
        LinkedList<String> files = new LinkedList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path path : stream) {
                if (Files.isRegularFile(path)) {
                    files.add(path.getFileName().toString());
                }
            }
        } catch (IOException | DirectoryIteratorException e) {
            System.err.println("Error listing files: " + e.getMessage());
        }
        return files;
    }

    public static Dataset readCleanDevign(String path) {
        Dataset dataset = new Dataset();
        List<String> files = listFiles(path);
        List<String> invalidFiles = new ArrayList<>();
        for (String file : files) {
            if (file.endsWith("0.c")) {
                String code = readFile(path + file);
                String header = extractMethodHeader(code);
                if (header == null) {
                    // System.out.println(file + "| "+ code + "header>>" + header);
                    invalidFiles.add(file);
                    continue;
                }
                dataset.addPair(file, header, code);
            }
        }
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("invalid_devign.txt"))) {
            for (String line : invalidFiles) {
                writer.write(line);
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        // System.out.println(dataset.getDataset().get(files.get(2)));
        return dataset;
    }

    @Deprecated
    public static Dataset readCleanBigvulCsv(String path) throws Exception {
        Dataset dataset = new Dataset();

        try (CSVReaderHeaderAware csvReader = new CSVReaderHeaderAware(new FileReader(path))) {
            Map<String, String> row;
            while ((row = csvReader.readMap()) != null) {
                String target = row.get("target");
                if (target != null && target.trim().equals("0")) {
                    String index = row.get("index");
                    String clean = row.get("processed_func");
                    dataset.addPair(index, extractMethodHeader(clean), clean);
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading CSV file: " + e.getMessage());
        }
        return dataset;
    }

    public static Dataset readFormattedVuls(String path) throws Exception {
        Dataset dataset = new Dataset();
        ObjectMapper objectMapper = new ObjectMapper();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                JsonNode jsonNode = objectMapper.readTree(line);
                JsonNode target = jsonNode.get("target");
                if (target != null && target.asInt() == 1) {
                    JsonNode index = jsonNode.get("index");
                    JsonNode vul = jsonNode.get("processed_func");
                    JsonNode flaw_lines = jsonNode.get("flaw_line");
                    if (index != null && vul != null && flaw_lines != null && flaw_lines.asText().trim().length() >= 5) {
                        String header = extractMethodHeader(vul.asText());
                        // if (header == null) {
                        //     System.out.println(index.asText() + "| "+ fixedVul.asText() + "header>>" + header);
                        // }

                        dataset.addPair(index.asText(), header, vul.asText());
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading CSV file: " + e.getMessage());
        }
        return dataset;
    }

    public static Dataset readCleanFromJson(String path) throws Exception {
        Dataset dataset = new Dataset();
        ObjectMapper objectMapper = new ObjectMapper();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                JsonNode jsonNode = objectMapper.readTree(line);
                JsonNode target = jsonNode.get("target");
                if (target != null && target.asInt() == 0) {
                    JsonNode index = jsonNode.get("index");
                    JsonNode clean = jsonNode.get("processed_func");
                    if (index != null && clean != null) {
                        dataset.addPair(index.asText(), extractMethodHeader(clean.asText()), clean.asText());
                    } else {
                        System.out.println("Wrong!");
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading JSON file: " + e.getMessage());
        }
        return dataset;
    }

    public static Dataset readCleanPrimeVul() throws Exception {
        return readCleanFromJson("../container_data/primevul_train_cleaned_complete.jsonl");
    }



    @Deprecated
    public static Dataset readFixedVulBigVulCsv(String path) throws Exception {
        Dataset dataset = new Dataset();

        try (CSVReaderHeaderAware csvReader = new CSVReaderHeaderAware(new FileReader(path))) {
            Map<String, String> row;
            while ((row = csvReader.readMap()) != null) {
                String target = row.get("target");
                if (target != null && target.trim().equals("1")) {
                    String index = row.get("index");
                    String fixedVul = row.get("func_after");
                    dataset.addPair(index, extractMethodHeader(fixedVul), fixedVul);
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading CSV file: " + e.getMessage());
        }
        return dataset;
    }

    public static Dataset readCleanBigvul() throws Exception {
        return readCleanFromJson("../container_data/bigvul-train.jsonl");
    }

    public static Dataset readFormattedVulsBigVul() throws Exception {
        return readFormattedVuls("../container_data/bigvul-train.jsonl");
    }

    public static Dataset readFormattedVulsPrimeVul() throws Exception {
        return readFormattedVuls("../container_data/primevul_train_cleaned_paired_full.jsonl");
    }

    public static Dataset readAllClean() throws Exception {
        Dataset devign = readCleanDevign();
        Dataset bigvul = readCleanBigvul();
        return Dataset.merge(devign, bigvul);
    }

    public static String extractMethodHeader(String fileContent) {
        int braceIndex = fileContent.indexOf('{');
        if (braceIndex != -1) {
            // Extract everything before the first '{'
            String header = fileContent.substring(0, braceIndex);
            // Optionally, remove any trailing whitespace or newlines
            header = header
                    .replaceAll("\\s+$", "")
                    .replaceAll("\n", " ")
                    .replaceAll(",\\s*", ", ")
                    .replaceAll("\\(\\s+", "(")
                    .replaceAll("s+\\)", "\\)")
                    .replaceAll("\\s+&", "& ")
                    .replaceAll("\\s+\\*", "* ")
                    .trim();

            return header;
        }
        return null;
    }

    public static String readFile(String filePath) {
        StringBuilder content = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                content.append(line).append("\n");
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
            return null;
        }
        return content.toString();
    }

}
