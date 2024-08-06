import json
from typing import List

# Define the classes to match the JSON structure


class ScoredCodeSnippet:
    def __init__(self, id: str, body: str, header: str, score: float, cluster: int):
        self.id = id
        self.body = body
        self.header = header
        self.score = score
        self.cluster = cluster

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        return (f"ScoredCodeSnippet(id='{self.id}', "
                # Print only the first 60 characters of the body for brevity
                f"body='{self.body[:60]}...', "
                f"header='{self.header}', "
                f"cluster='{self.cluster}', "
                f"score={self.score})")


class SearchItem:
    def __init__(self, searchItemId: str, scoredCodeSnippets: List[ScoredCodeSnippet]):
        self.searchItemId = searchItemId
        self.scoredCodeSnippets = scoredCodeSnippets
        self.best, self.max_score, self.avg_score = self.__find_max(scoredCodeSnippets) 

    def __find_max(self, snippets: List[ScoredCodeSnippet]):
        avg = 0
        max_score = -1
        best_match = None
        for snippet in snippets:
            avg += snippet.score /len(snippets)
            if snippet.score > max_score:
                max_score = snippet.score
                best_match = snippet.id
        return best_match, max_score, avg

    def get_flattened(self):
        items = []
        for snippet in self.scoredCodeSnippets:
            newItem = SearchItem(self.searchItemId, [snippet])
            items.append(newItem)
        return items

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        snippets_str = "\n  ".join(str(snippet)
                                   for snippet in self.scoredCodeSnippets)
        return (f"SearchItem(searchItemId='{self.searchItemId}', "
                f"scoredCodeSnippets=[\n  {snippets_str}\n])")


def flatten_search_items(search_results: List[SearchItem]):
    items = []
    for result in search_results:
        for item in result.get_flattened():
            items.append(item)
    return items    

def json_to_classes(json_data):
    search_items = []
    for item in json_data:
        searchItemId = item["searchItemId"]
        scoredCodeSnippets = []
        for snippet in item["scoredCodeSnippets"]:
            cluster = -1
            if "clusterIndex" in snippet:
                cluster = int(snippet["clusterIndex"])
            scoredCodeSnippets.append(
                ScoredCodeSnippet(
                    id=snippet["id"],
                    body=snippet["body"],
                    header=snippet["header"],
                    score=float(snippet["score"]),
                    cluster=cluster
                )
            )
        search_items.append(SearchItem(searchItemId, scoredCodeSnippets))
    return search_items


def read_rag(file_name):
    result = None
    with open(f"./RAG/{file_name}.json", "r") as file:
        data = json.load(file)
        result = json_to_classes(data)

    return result

def read_naive_code2code_ext_rag():
    return read_rag("results_full_code2code_ext")

def read_naive_code2code_rag():
    return read_rag("results_full_code2code")

def read_naive_code2code_clustered_rag():
    return read_rag("results_full_code2code_clustered")

def read_random():
    return read_rag("results_random")

def read_random_fair():
    return read_rag("results_random_fair")

def read_naive_full_header_rag():
    return read_rag("results_full_header")


def read_naive_full_bigvul_header_rag():
    return read_rag("results_full_header_bigvul")

def read_naive_full_func_rag():
    return read_rag("results_full_func")