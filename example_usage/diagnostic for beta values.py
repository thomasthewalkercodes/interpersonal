"""
INSPECT BETA FILE CONTENTS
===========================
This script will compare the contents of β=5.0, β=10.0, and β=15.0 files
to see why the analysis script only loads β=5.0 files.
"""

import os
import json
import glob


def find_sample_directories():
    """Find sample directories for each beta value"""
    print("🔍 FINDING SAMPLE DIRECTORIES FOR EACH BETA VALUE")
    print("=" * 60)

    if not os.path.exists("results"):
        print("❌ No results directory found")
        return None, None, None

    all_dirs = os.listdir("results")

    # Find one directory for each beta value
    beta_5_sample = None
    beta_10_sample = None
    beta_15_sample = None

    for d in all_dirs:
        if "beta_5" in d or "beta_5.0" in d:
            if beta_5_sample is None:
                beta_5_sample = d
        elif "beta_10" in d or "beta_10.0" in d:
            if beta_10_sample is None:
                beta_10_sample = d
        elif "beta_15" in d or "beta_15.0" in d:
            if beta_15_sample is None:
                beta_15_sample = d

    print(f"📂 Sample directories found:")
    print(f"   β=5.0:  {beta_5_sample}")
    print(f"   β=10.0: {beta_10_sample}")
    print(f"   β=15.0: {beta_15_sample}")

    return beta_5_sample, beta_10_sample, beta_15_sample


def inspect_directory_contents(directory_name, beta_label):
    """Inspect the contents of a specific directory"""
    if not directory_name:
        print(f"\n❌ No {beta_label} directory to inspect")
        return None

    print(f"\n🔍 INSPECTING {beta_label} DIRECTORY: {directory_name}")
    print("-" * 50)

    full_path = os.path.join("results", directory_name)

    if not os.path.exists(full_path):
        print(f"❌ Directory doesn't exist: {full_path}")
        return None

    # List all files in directory
    files = os.listdir(full_path)
    print(f"📁 Files in directory: {files}")

    results = {}

    # Check results.json
    results_json_path = os.path.join(full_path, "results.json")
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, "r") as f:
                data = json.load(f)

            print(f"\n📄 results.json contents:")
            print(f"   File size: {os.path.getsize(results_json_path)} bytes")
            print(f"   Data type: {type(data)}")

            if isinstance(data, dict):
                print(f"   Top-level keys: {list(data.keys())}")

                # Look for final_evaluation
                if "final_evaluation" in data:
                    final_eval = data["final_evaluation"]
                    print(f"   final_evaluation type: {type(final_eval)}")

                    if isinstance(final_eval, dict):
                        print(f"   final_evaluation keys: {list(final_eval.keys())}")

                        # Check the specific values we need
                        agent1_reward = final_eval.get("agent1_avg_reward")
                        agent2_reward = final_eval.get("agent2_avg_reward")

                        print(
                            f"   agent1_avg_reward: {agent1_reward} (type: {type(agent1_reward)})"
                        )
                        print(
                            f"   agent2_avg_reward: {agent2_reward} (type: {type(agent2_reward)})"
                        )

                        # Try to convert to float (this is where the analysis might be failing)
                        try:
                            agent1_float = (
                                float(agent1_reward)
                                if agent1_reward is not None
                                else None
                            )
                            agent2_float = (
                                float(agent2_reward)
                                if agent2_reward is not None
                                else None
                            )
                            print(
                                f"   ✅ Conversion successful: {agent1_float}, {agent2_float}"
                            )
                        except Exception as e:
                            print(f"   ❌ Conversion failed: {e}")
                    else:
                        print(f"   ⚠️ final_evaluation is not a dict: {final_eval}")
                else:
                    print(f"   ⚠️ No 'final_evaluation' key found")
                    print(f"   Available keys: {list(data.keys())}")

                # Look for config information
                if "config" in data:
                    config = data["config"]
                    print(f"   config type: {type(config)}")
                    if isinstance(config, dict):
                        alpha = config.get("payoff_alpha", config.get("alpha"))
                        beta = config.get("payoff_beta", config.get("beta"))
                        print(f"   config alpha: {alpha}")
                        print(f"   config beta: {beta}")

            results["results_json"] = data

        except Exception as e:
            print(f"   ❌ Error reading results.json: {e}")
    else:
        print(f"\n⚠️ No results.json found")

    # Check config.json
    config_json_path = os.path.join(full_path, "config.json")
    if os.path.exists(config_json_path):
        try:
            with open(config_json_path, "r") as f:
                config_data = json.load(f)

            print(f"\n📄 config.json contents:")
            print(f"   File size: {os.path.getsize(config_json_path)} bytes")
            print(f"   Data type: {type(config_data)}")

            if isinstance(config_data, dict):
                print(f"   Keys: {list(config_data.keys())}")
                alpha = config_data.get("payoff_alpha", config_data.get("alpha"))
                beta = config_data.get("payoff_beta", config_data.get("beta"))
                print(f"   alpha: {alpha}")
                print(f"   beta: {beta}")

            results["config_json"] = config_data

        except Exception as e:
            print(f"   ❌ Error reading config.json: {e}")
    else:
        print(f"\n⚠️ No config.json found")

    return results


def compare_file_structures(beta_5_data, beta_10_data, beta_15_data):
    """Compare the structures of different beta files"""
    print(f"\n🔍 COMPARING FILE STRUCTURES")
    print("=" * 60)

    for beta_name, data in [
        ("β=5.0", beta_5_data),
        ("β=10.0", beta_10_data),
        ("β=15.0", beta_15_data),
    ]:
        if data is None:
            print(f"\n❌ No data for {beta_name}")
            continue

        print(f"\n📊 {beta_name} Structure:")

        if "results_json" in data:
            results_json = data["results_json"]
            if isinstance(results_json, dict):
                print(f"   results.json keys: {list(results_json.keys())}")

                if "final_evaluation" in results_json:
                    final_eval = results_json["final_evaluation"]
                    if isinstance(final_eval, dict):
                        agent1 = final_eval.get("agent1_avg_reward")
                        agent2 = final_eval.get("agent2_avg_reward")
                        print(f"   agent1_avg_reward: {agent1} ({type(agent1)})")
                        print(f"   agent2_avg_reward: {agent2} ({type(agent2)})")

                        # This is the critical check - can we convert to float?
                        try:
                            if agent1 is not None and agent2 is not None:
                                float(agent1)
                                float(agent2)
                                print(f"   ✅ Values can be converted to float")
                            else:
                                print(
                                    f"   ❌ Values are None - this will cause loading to fail"
                                )
                        except:
                            print(
                                f"   ❌ Values cannot be converted to float - this will cause loading to fail"
                            )
                    else:
                        print(
                            f"   ❌ final_evaluation is not a dict - this will cause loading to fail"
                        )
                else:
                    print(
                        f"   ❌ No final_evaluation key - this will cause loading to fail"
                    )
            else:
                print(
                    f"   ❌ results.json is not a dict - this will cause loading to fail"
                )
        else:
            print(f"   ❌ No results.json - this will cause loading to fail")


def test_analysis_script_loading(beta_10_sample, beta_15_sample):
    """Test exactly what the analysis script would do with these files"""
    print(f"\n🧪 TESTING ANALYSIS SCRIPT LOADING LOGIC")
    print("=" * 60)

    if not beta_10_sample:
        print("❌ No β=10.0 sample to test")
        return

    # Simulate exactly what the analysis script does
    personalities = ["cooperative", "competitive", "adaptive", "cautious"]
    alphas = [2.0, 4.0, 6.0]
    betas = [5.0, 10.0, 15.0]

    print("🔍 Testing loading logic for β=10.0 files...")

    loaded_count = 0
    failed_count = 0

    for agent1_type in personalities:
        for agent2_type in personalities:
            for alpha in alphas:
                for beta in [10.0]:  # Just test β=10.0
                    filename_pattern = (
                        f"sweep_{agent1_type}_{agent2_type}_alpha_{alpha}_beta_{beta}"
                    )
                    filepath = f"results/{filename_pattern}/results.json"

                    if os.path.exists(filepath):
                        try:
                            with open(filepath, "r") as f:
                                result = json.load(f)

                            if (
                                isinstance(result, dict)
                                and "final_evaluation" in result
                            ):
                                final_eval = result["final_evaluation"]

                                # This is where the analysis script could be failing
                                try:
                                    agent1_reward = float(
                                        final_eval["agent1_avg_reward"]
                                    )
                                    agent2_reward = float(
                                        final_eval["agent2_avg_reward"]
                                    )
                                    loaded_count += 1
                                    print(
                                        f"   ✅ {filename_pattern}: {agent1_reward}, {agent2_reward}"
                                    )
                                except Exception as e:
                                    failed_count += 1
                                    print(
                                        f"   ❌ {filename_pattern}: Conversion error - {e}"
                                    )
                                    print(
                                        f"      agent1_avg_reward: {final_eval.get('agent1_avg_reward')}"
                                    )
                                    print(
                                        f"      agent2_avg_reward: {final_eval.get('agent2_avg_reward')}"
                                    )
                            else:
                                failed_count += 1
                                print(f"   ❌ {filename_pattern}: Structure error")
                        except Exception as e:
                            failed_count += 1
                            print(f"   ❌ {filename_pattern}: File error - {e}")
                    else:
                        print(f"   ⚠️ {filename_pattern}: File not found")

    print(f"\n📊 Loading Test Results:")
    print(f"   ✅ Successfully loaded: {loaded_count}")
    print(f"   ❌ Failed to load: {failed_count}")

    if failed_count > 0:
        print(f"\n💡 The analysis script is failing because:")
        print(f"   - Files have different structure than expected")
        print(f"   - Values cannot be converted to numbers")
        print(f"   - Missing required keys in the JSON")


def main():
    """Main inspection function"""
    print("🔍 BETA FILE CONTENTS INSPECTOR")
    print("=" * 60)

    # Step 1: Find sample directories
    beta_5_sample, beta_10_sample, beta_15_sample = find_sample_directories()

    # Step 2: Inspect each directory
    beta_5_data = inspect_directory_contents(beta_5_sample, "β=5.0")
    beta_10_data = inspect_directory_contents(beta_10_sample, "β=10.0")
    beta_15_data = inspect_directory_contents(beta_15_sample, "β=15.0")

    # Step 3: Compare structures
    compare_file_structures(beta_5_data, beta_10_data, beta_15_data)

    # Step 4: Test the analysis script loading logic
    test_analysis_script_loading(beta_10_sample, beta_15_sample)

    print(f"\n🎯 NEXT STEPS:")
    print("=" * 60)
    print("Based on the inspection results above, I can now:")
    print("1. ✅ Fix the analysis script if there's a structure difference")
    print("2. ✅ Identify what's causing the loading to fail")
    print("3. ✅ Modify the file parsing logic to handle your specific format")


if __name__ == "__main__":
    main()
