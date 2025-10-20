MAGIC_CODER_TEMPLATE_SYS = """
    You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions.
    """

MAGIC_CODER_TEMPLATE_USER = """
    Please gain inspiration from the following random code snippet to create a high-quality programming problem. Present your output in two distinct sections: [Problem Description] and [Solution].
    
    Code snippet for inspiration: 
    ```  
    {code} 
    ```  
    
    Guidelines for each section:  1. [Problem Description]: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included. 2. [Solution]: Offer a comprehensive, **correct** solution that accurately addresses the [Problem Description] you provided.
    
    """
    
CLAUDE_TEMPLATE_SYS = """
    You are an expert at creating well-structured, educational programming problems inspired by code snippets. You excel at identifying core concepts and transforming them into clear, engaging challenges with comprehensive solutions.
    """
    
CLAUDE_TEMPLATE_USER = """
    Using the following code snippet as inspiration, create a high-quality programming problem. Your response must be in valid JSON format with the following structure:
    
    Code snippet for inspiration:
    ```
    {code}
    ```
    
    Return a JSON object with this exact structure:
    {{
        "problem": "string - complete problem description including all constraints, input/output format, and examples",
        "solution": "string - complete working solution code with comments, complexity analysis, and test cases"
    }}
    
    Requirements:
    ## Problem Requirements:
    - Must be completely self-contained and require no external context
    - Include all necessary information, constraints, and examples
    - Clearly state the expected input/output format
    - Provide 2-3 concrete examples with edge cases
    - Specify any performance requirements or constraints
    - Use clear, unambiguous language
    
    ## Solution Requirements:
    - Provide a complete, working solution that solves the problem exactly as described
    - Include time and space complexity analysis
    - Add meaningful comments explaining the approach
    - Consider multiple approaches if applicable
    - Include test cases that validate the solution
    - Ensure the solution handles all edge cases mentioned in the problem
    
    Additional Guidelines:
    - The problem should teach a specific concept or technique evident in the snippet
    - Difficulty should be appropriate for technical interviews or competitive programming
    - Avoid trivial problems; ensure there's a meaningful algorithmic challenge
    - The problem description should stand alone - someone should be able to solve it without seeing the inspiration code
    
    Ensure your response is valid JSON that can be parsed directly.
    """