##  Text-to-Knowledge Graph with LLM framework  

[Colab link](https://colab.research.google.com/drive/1_Y50l0QXpj-WRQQsQCijHeuxnOUzFevZ?usp=sharing)

## Explain each part  
### 0. setup environment    
#### Install and import some libraries and functions  
code in this part is mainly the modified version of [this github repository](https://github.com/rahulnyk/knowledge_graph)  I rearrange some parts hoping for more simplicity, and delete some parts that we didn't use.  

here are some important functions for this project.

df2Graph will be used in 3.2 to make an concept_list, which contained all triplets in , extracted from joke and chosen model
```python
def df2Graph(dataframe: pd.DataFrame, model=None) -> list:
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, {"chunk_id": row.chunk_id}, model), axis=1
    )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list
```

used in 3.3 to store triplets from knowledge graph in dataframe format 
```python
def graph2Df(nodes_list) -> pd.DataFrame:
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())

    return graph_dataframe
```

____after some modification, this function will not do anything for now. It will be developed into a function to handle multiple texts.___
```python
def text2Dataframe(text: str) -> pd.DataFrame:
    # Instantiate the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )

    # Split the text into chunks
    chunks = splitter.split_text(text)

    # Create rows for each chunk
    rows = []
    for chunk_content in chunks:
        row = {
            "text": chunk_content,
            "chunk_id": uuid.uuid4().hex,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
``` 

set up an output directory to store output data.
```python
## This is where the output csv files will be written
out_dir = "here"
outputdirectory = Path(f"./data_output/{out_dir}")
```
I installed the older version of OpenAI because the newer version has combability issue **(must add more details)**. 
#### setup Gemini & OpenAI  
insert API keys for your preferred models. 

code for import google generative AI, set up API key and model's safety sitting 
```python 
import google.generativeai as genai 

os.environ["GOOGLE_AI_API_KEY"] = 'your_key'

genai.configure(api_key=os.environ["GOOGLE_AI_API_KEY"])

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  }
]

model_G = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
```

code for import OpenAI and setup OpenAI's API key
```python 
import openai
# Set OpenAI API key
openai.api_key = 'your_key'
```
### 2. Get joke text   
we can add joke text from the prepared dataset or add it manually. 
### 3. transform text from str to dataframe format 
performs text to knowledge graph 
#### 3.1 select our prompt 
there are many ways to prompt the model to generate Knowledge Graph
we will use the same format as [this project](https://github.com/rahulnyk/knowledge_graph)which can be simplified into this structure.
```python
SYS_PROMPT = (
    "text to tell the model on how to Generate Knowledge Graph in json format \n"
)

USER_PROMPT = f"input text: {textfortest} \n\n output: "
prompt_text=f"{SYS_PROMPT}\n\n{USER_PROMPT}"
```
then we will experiment with many prompts to find the suitable one. 

the example from many prompts:
```python
SYS_PROMPT = (
    "Generate Knowledge Graph from the input text in the provided format: \n"
    "[\n"
    "   {\n"
    '       "node_1": "1st term",\n'
    '       "node_2": "2nd term",\n'
    '       "edge": "Relationship between node_1 and node_2 in one or two sentences"\n'
    "   }, {...}\n"
    "]"
)

USER_PROMPT = f"input text: {textfortest} \n\n output: "
prompt_text=f"{SYS_PROMPT}\n\n{USER_PROMPT}"
```

#### 3.2 select our LLM model and generate Knowledge Graph 
each model has a different way to prompt in small details.

OpenAI's model
```python
def graphPrompt(input: str, metadata={}, model="gpt-3.5-turbo-instruct"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt_text,
        max_tokens=1000 
    )
    try:
        result = json.loads(response['choices'][0]['text'])
        result = [dict(item, **metadata) for item in result]
    except Exception as e:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result
# get concept list from our prompt and model
concepts_list = df2Graph(df, model="gpt-3.5-turbo-instruct")
```

Gemini model
```python
def graphPrompt(input: str, metadata={}, model=model_G):
    response = model.generate_content(prompt_text)
    try:
        result = json.loads(response.text)
        result = [dict(item, **metadata) for item in result]
    except Exception as e:
        print("\n\nERROR ### Here is the buggy response: ", response.text, "\n\n")
        result = None
    return result 
# get concept list from our prompt and model    
concepts_list = df2Graph(df, model=genai.GenerativeModel('gemini-pro', safety_settings=safety_settings))
```
#### 3.3 store Knowledge Graph in dataframe and .csv format
JSON --> dataframe & .csv 
we will use graph2Df function to store KG in dataframe format, then store .csv file in determined output directory.
```python
## To regenerate the graph with LLM, set this to True
regenerate = True
if regenerate:
  # get concept list from our prompt and model
    dfg1 = graph2Df(concepts_list)
    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)

    dfg1.to_csv(outputdirectory/"graph.csv", sep="|", index=False)
    df.to_csv(outputdirectory/"chunks.csv", sep="|", index=False)
else:
    dfg1 = pd.read_csv(outputdirectory/"graph.csv", sep="|")

dfg1.replace("", np.nan, inplace=True)
dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
dfg1['count'] = 4
## Increasing the weight of the relation to 4.
## We will assign the weight of 1 when later the contextual proximity will be calculated.
print(dfg1.shape)
dfg1.head()
```
### 4. (optional) calculate contextual proximity  
I think this function might not suitable for our use case because it makes our knowledge graph a fully connected graph in many cases. but It might be useful for analyzing long texts.
#### 4.1 Original Graph from prompt (recommend)
I recommend to just run this one and skip to the 5th part. 
```python
dfg = dfg1
```
#### 4.2 Graph from prompt + contextual proximity
this code will make another dataframe by calculating the contextual proximity between each node, then merge this dataframe with the dataframe from 3.3

### 5. graph visualize  
 as its name suggests. we will coloring each node according to its group, then output in HTML format  
## Remarks
- prompts in the 3rd part are still not suitable for extract implicit relationship
- ways to improve KG generator
	- Change SYS_PROMPT
	- Rearrange SYS_PROMPT and USER_PROMPT' order in 3.1 part. sometime I found it affect the output, but this is just a hypothesis.
	- Change the LLM model.
