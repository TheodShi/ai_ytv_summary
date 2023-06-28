import os
from dotenv import load_dotenv, find_dotenv

# Import Azure OpenAI
from langchain.llms import AzureOpenAI
from langchain.llms import OpenAI
from langchain import PromptTemplate

class LLM:
    __delimiter__ = "####"
    __prompt__ = PromptTemplate(
        input_variables=["delimiter", "speech"],
        template="""
            请根据下面的文本生成一个中文摘要，文本的内容被包含在{delimiter}分隔符内.
            生成的摘要请遵循下面的条件:
                a. 不要包含打招呼和感谢的内容
                b. 生成的摘要内容要尽可能清晰,简洁
                c. 文本如果包含多个结论,请使用a,b,c这样的方式去分别罗列每个结论
            
            {delimiter}
            {speech}
            {delimiter}
        """,
    )

    def __init__(self) -> None:
        _ = load_dotenv(find_dotenv()) # read local .env file

        match (os.environ['AI_API_TYPE']):
            case "openai":
                self.llm = OpenAI(
                    openai_api_key=os.environ['OPENAI_API_KEY'],
                    temperature=0.0,
                )
            case "azure":
                self.llm = AzureOpenAI(
                    openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],
                    openai_api_base=os.environ['AZURE_OPENAI_API_BASE'],
                    openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
                    deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
                    model_name=os.environ['AZURE_OPENAI_MODEL_NAME'],
                    temperature=0.0,
                )
                
        if self.llm is None:
            raise Exception("LLM initialization failed...")

    def generate_summary(self, text):
        llm_result = self.llm.generate([self.__prompt__.format(delimiter=self.__delimiter__, speech=text)])

        print(f"token_usage: {llm_result.llm_output['token_usage']['total_tokens']}, Approximately cost: ${round(0.02 * llm_result.llm_output['token_usage']['total_tokens'] / 1000, 2)}")
        
        return llm_result.generations[0][0].text

if (__name__ == '__main__'):
    llm = LLM()
    response = llm.generate_summary("""
        大语言模型 (英语：large language model，LLM) 是一种语言模型，由具有许多参数（通常数十亿个权重或更多）的人工神经网络组成，使用自监督学习或半监督学习对大量未标记文本进行训练[1]。大型语言模型在2018年左右出现，并在各种任务中表现出色[2]。

        尽管这个术语没有正式的定义，但它通常指的是参数数量在数十亿或更多数量级的深度学习模型[3]。大型语言模型是通用的模型，在广泛的任务中表现出色，而不是针对一项特定任务（例如情感分析、命名实体识别或数学推理）进行训练[2]。

        尽管在预测句子中的下一个单词等简单任务上接受过训练，但发现具有足够训练和参数计数的神经语言模型可以捕获人类语言的大部分句法和语义。 此外大型语言模型展示了相当多的关于世界的常识，并且能够在训练期间“记住”大量事实[2]。
        """
    )

    print(response)
