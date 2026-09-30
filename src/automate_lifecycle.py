import mlflow
from mlflow.tracking import MlflowClient


def automate_champion_challenger():
    model_name = "Employee_Attrition_Production_Model"
    metric_to_optimize = "f1"

    client = MlflowClient()

    print("=" * 70)
    print("EMPLOYEE ATTRITION CHAMPION-CHALLENGER LIFECYCLE")
    print("=" * 70)

    try:
        versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        if not versions:
            print(f"[ERROR] No registered versions found for {model_name}")
            return

        print(f"[INFO] Registered model: {model_name}")
        print(f"[INFO] Total versions found: {len(versions)}")

        staging_versions = []

        for version in versions:
            if version.current_stage == "None":
                try:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version.version,
                        stage="Staging",
                        archive_existing_versions=False
                    )

                    print(
                        f"[INFO] Version {version.version} "
                        f"moved to Staging"
                    )

                except Exception as exc:
                    print(
                        f"[WARNING] Could not move version "
                        f"{version.version} to Staging: {exc}"
                    )

        versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        for version in versions:
            if version.current_stage == "Staging":
                staging_versions.append(version)

        if not staging_versions:
            print("[ERROR] No Staging versions available")
            return

        print(
            f"[INFO] Staging versions: "
            f"{[v.version for v in staging_versions]}"
        )

        candidate_scores = []

        for version in staging_versions:
            run_id = version.run_id

            if not run_id:
                print(
                    f"[WARNING] Version {version.version} "
                    f"has no run ID"
                )
                continue

            run = client.get_run(run_id)

            metric_value = run.data.metrics.get(metric_to_optimize)

            if metric_value is None:
                print(
                    f"[WARNING] Version {version.version} "
                    f"does not contain metric '{metric_to_optimize}'"
                )
                continue

            candidate_scores.append(
                {
                    "version": version.version,
                    "run_id": run_id,
                    "score": metric_value
                }
            )

            print(
                f"[INFO] Version {version.version} "
                f"{metric_to_optimize}={metric_value:.4f}"
            )

        if not candidate_scores:
            print("[ERROR] No valid Challenger models found")
            return

        candidate_scores.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        challenger = candidate_scores[0]

        production_versions = [
            version
            for version in versions
            if version.current_stage == "Production"
        ]

        champion = None

        if production_versions:
            champion_version = production_versions[0]

            if champion_version.run_id:
                champion_run = client.get_run(
                    champion_version.run_id
                )

                champion_score = champion_run.data.metrics.get(
                    metric_to_optimize
                )

                if champion_score is not None:
                    champion = {
                        "version": champion_version.version,
                        "run_id": champion_version.run_id,
                        "score": champion_score
                    }

        print("-" * 70)

        if champion is None:
            print("[INFO] No current Production Champion found")
            print(
                f"[INFO] Challenger version "
                f"{challenger['version']} will be promoted"
            )

            client.transition_model_version_stage(
                name=model_name,
                version=challenger["version"],
                stage="Production",
                archive_existing_versions=False
            )

            print(
                f"[SUCCESS] Version {challenger['version']} "
                f"promoted to Production"
            )

        else:
            print(
                f"[INFO] Champion version: "
                f"{champion['version']}"
            )

            print(
                f"[INFO] Champion {metric_to_optimize}: "
                f"{champion['score']:.4f}"
            )

            print(
                f"[INFO] Challenger version: "
                f"{challenger['version']}"
            )

            print(
                f"[INFO] Challenger {metric_to_optimize}: "
                f"{challenger['score']:.4f}"
            )

            if challenger["score"] > champion["score"]:

                print(
                    "[INFO] Challenger has a higher "
                    f"{metric_to_optimize}"
                )

                client.transition_model_version_stage(
                    name=model_name,
                    version=challenger["version"],
                    stage="Production",
                    archive_existing_versions=True
                )

                print(
                    f"[SUCCESS] Challenger version "
                    f"{challenger['version']} promoted to Production"
                )

                print(
                    f"[INFO] Previous Champion version "
                    f"{champion['version']} archived"
                )

            else:

                print(
                    "[INFO] Challenger did not exceed "
                    "the current Champion"
                )

                client.transition_model_version_stage(
                    name=model_name,
                    version=challenger["version"],
                    stage="Archived"
                )

                print(
                    f"[INFO] Challenger version "
                    f"{challenger['version']} archived"
                )

        for candidate in candidate_scores[1:]:
            version_number = candidate["version"]

            try:
                current_version = client.get_model_version(
                    name=model_name,
                    version=version_number
                )

                if current_version.current_stage == "Staging":
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version_number,
                        stage="Archived"
                    )

                    print(
                        f"[INFO] Non-selected Staging version "
                        f"{version_number} archived"
                    )

            except Exception as exc:
                print(
                    f"[WARNING] Could not archive version "
                    f"{version_number}: {exc}"
                )

        print("-" * 70)
        print("FINAL MODEL LIFECYCLE STATUS")
        print("-" * 70)

        final_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        for version in sorted(
            final_versions,
            key=lambda item: int(item.version)
        ):
            print(
                f"Version {version.version}: "
                f"{version.current_stage}"
            )

        print("=" * 70)
        print("[SUCCESS] Champion-Challenger lifecycle completed")
        print("=" * 70)

    except Exception as exc:
        print(
            f"[ERROR] Champion-Challenger lifecycle failed: {exc}"
        )


if __name__ == "__main__":
    automate_champion_challenger()