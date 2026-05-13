import os
import sys
import subprocess
import re
from google import genai

class SovereignBrain:
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("❌ ERROR: GEMINI_API_KEY environment variable missing.")
            sys.exit(1)
            
        print("🔌 Initializing Gemini 2.5 Pro via Unified SDK...")
        self.client = genai.Client(api_key=self.api_key)
        print("🤖 System Online. Workspace connected.")

    def extract_python_code(self, llm_response: str) -> str:
        """
        Lobotomy Layer: Extracts pure Python code from within markdown blocks.
        Strips away all conversational descriptions.
        """
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, llm_response, re.DOTALL)
        
        if match:
            print("⚡ REGEX: Successfully extracted pure code block.")
            return match.group(1).strip()
        
        print("⚠️ REGEX WARNING: No markdown block found. Processing raw text.")
        return llm_response.strip()

    def run_autonomous_remediation(self, log_filename: str):
        """
        Step 1: Ingest the raw crash telemetry log.
        """
        log_path = os.path.join(self.workspace_path, "src/sovereign_core", log_filename)
        print(f"📖 Ingesting telemetry data from: {log_path}")
        
        try:
            with open(log_path, "r") as f:
                crash_data = f.read()
        except FileNotFoundError:
            print(f"❌ Error log not found at {log_path}")
            return

        # Step 2: Construct full context prompt
        prompt = f"""
        You are an Autonomous Systems Engineer (Sovereign-CI).
        
        CRASH TRACEBACK DATA:
        {crash_data}
        
        TASK:
        1. Identify the root dimensional conflict in openfold/model/embedders.py.
        2. Provide ONLY a valid, complete Python class or module replacement block that updates the self.linear_tf_z_i layer initialization to handle 256 in_features instead of 128.
        
        OUTPUT FORMAT:
        You must encapsulate your code fix inside standard markdown code fences like this:
        ```python
        # Code here
        ```
        """

        print("🧠 Uploading problem matrix to Gemini 2.5 Pro for analysis...")
        response = self.client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt
        )
        
        print("\n--- 🔵 GEMINI REASONING MATRIX ---")
        print(response.text)
        print("----------------------------------\n")
        
        # Step 3: Pass live model output directly into the staging engine
        self.stage_mcp_patch_for_review(response.text)

    def stage_mcp_patch_for_review(self, ai_output: str):
        """
        Step 4: Clean the output and stage it for repository modification.
        """
        # Dynamically clean the AI's output using the regex engine
        clean_code = self.extract_python_code(ai_output)
        
        print("⚡ STAGING PHASE: Initiating local patch isolate...")
        target_file = "src/sovereign_core/openfold/model/embedders.py"
        target_path = os.path.join(self.workspace_path, target_file)

        # Write the dynamically generated code directly to disk
        with open(target_path, "w") as f:
            f.write(clean_code)
            
        print(f"🛠️  File modifications applied locally to: {target_file}")
        print("\n🔍 --- TARGET CODE MODIFICATIONS (GIT DIFF) ---")
        
        # Display the real modifications to the user via a git diff
        subprocess.run(["git", "--no-paper", "diff", target_file], cwd=self.workspace_path)
        print("------------------------------------------------\n")
        
        # Step 5: User verification prompt
        user_choice = input("⚠️  Review the diff above. Do you want to commit these changes to your branch history? (yes/no): ")
        
        if user_choice.lower() in ["yes", "y"]:
            print("🚀 Committing changes via local repository engine...")
            # Explicitly stage the file so Git registers the change for the commit
            subprocess.run(["git", "add", target_file], cwd=self.workspace_path)
            subprocess.run(["git", "commit", "-m", "fix(ops): dynamically adjust matrix dimensions inside embedder pipeline"], cwd=self.workspace_path)
            print("🏁 STATUS: Fix successfully committed to your branch history.")
        else:
            print("🛑 Operations paused by user. Modifications remain uncommitted.")

if __name__ == "__main__":
    # Absolute system workspace anchor
    engine = SovereignBrain(workspace_path="/Users/admin_m3_1/Desktop/sovereign_project")
    engine.run_autonomous_remediation("openfold_crash.log")
