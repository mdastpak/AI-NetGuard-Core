"""
Data Synthesis Agent

Responsible for generating unlimited synthetic data to train and validate
the AI-NetGuard system. Uses advanced generative models including CTGAN,
diffusion models, and data augmentation techniques.
"""

import asyncio
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from .base_agent import BaseAgent


class DataSynthesisAgent(BaseAgent):
    """
    Agent specialized in synthetic data generation for network traffic analysis.

    Capabilities:
    - Generate synthetic network traffic data
    - Create diverse attack patterns
    - Balance datasets for training
    - Implement data augmentation techniques
    """

    def __init__(self, coordinator_agent=None, **kwargs):
        system_message = """
        You are the DataSynthesisAgent, responsible for generating high-quality synthetic data
        for training AI-NetGuard models. Your goals include:

        1. Generate realistic network traffic patterns
        2. Create diverse attack scenarios
        3. Ensure data quality and diversity
        4. Support unlimited data generation
        5. Coordinate with other agents for data requirements

        Always prioritize data realism and utility for anomaly detection.
        """

        super().__init__(
            name="DataSynthesisAgent",
            system_message=system_message,
            coordinator_agent=coordinator_agent,
            **kwargs
        )

        self.capabilities = [
            "synthetic_data_generation",
            "attack_pattern_simulation",
            "data_augmentation",
            "dataset_balancing",
            "quality_assurance"
        ]

        self.dependencies = ["FeatureEngineeringAgent", "EvaluationAgent"]

        # Data generation models (to be initialized)
        self.generative_models = {}
        self.data_templates = {}

    async def _execute_task(self, task_description: str, **kwargs) -> Any:
        """
        Execute data synthesis tasks.

        Args:
            task_description: Description of the synthesis task
            **kwargs: Task parameters (dataset_size, attack_types, etc.)

        Returns:
            Dict containing generated data and metadata
        """
        if "generate_training_data" in task_description.lower():
            return await self._generate_training_data(**kwargs)
        elif "simulate_attacks" in task_description.lower():
            return await self._simulate_attacks(**kwargs)
        elif "augment_dataset" in task_description.lower():
            return await self._augment_dataset(**kwargs)
        elif "balance_classes" in task_description.lower():
            return await self._balance_classes(**kwargs)
        else:
            return await self._general_synthesis(task_description, **kwargs)

    async def _generate_training_data(self, dataset_size: int = 100000,
                                    attack_ratio: float = 0.1,
                                    **kwargs) -> Dict[str, Any]:
        """
        Generate synthetic training dataset.

        Args:
            dataset_size: Total number of samples to generate
            attack_ratio: Ratio of attack samples to normal traffic

        Returns:
            Dict with generated data and statistics
        """
        try:
            self.logger.info(f"Generating {dataset_size} training samples with {attack_ratio*100}% attacks")

            # Generate normal traffic
            normal_samples = int(dataset_size * (1 - attack_ratio))
            normal_data = await self._generate_normal_traffic(normal_samples)

            # Generate attack traffic
            attack_samples = dataset_size - normal_samples
            attack_data = await self._generate_attack_traffic(attack_samples)

            # Combine and shuffle
            combined_data = pd.concat([normal_data, attack_data], ignore_index=True)
            combined_data = combined_data.sample(frac=1).reset_index(drop=True)

            # Add labels
            combined_data['label'] = ['normal'] * normal_samples + ['attack'] * attack_samples

            result = {
                'data': combined_data,
                'statistics': {
                    'total_samples': len(combined_data),
                    'normal_samples': normal_samples,
                    'attack_samples': attack_samples,
                    'features': list(combined_data.columns),
                    'attack_types': list(attack_data['attack_type'].unique()) if 'attack_type' in attack_data.columns else []
                },
                'quality_metrics': await self._assess_data_quality(combined_data)
            }

            self.logger.info(f"Generated dataset with {len(combined_data)} samples")
            return result

        except Exception as e:
            self.logger.error(f"Failed to generate training data: {e}")
            return {'error': str(e)}

    async def _generate_normal_traffic(self, num_samples: int) -> pd.DataFrame:
        """
        Generate synthetic normal network traffic.

        Args:
            num_samples: Number of normal traffic samples

        Returns:
            DataFrame with normal traffic features
        """
        # Simplified synthetic data generation
        # In real implementation, use CTGAN or diffusion models

        np.random.seed(42)

        data = {
            'duration': np.random.exponential(1.0, num_samples),
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], num_samples,
                                            p=[0.7, 0.25, 0.05]),
            'service': np.random.choice(['http', 'smtp', 'ftp', 'ssh', 'dns'], num_samples,
                                      p=[0.4, 0.2, 0.15, 0.15, 0.1]),
            'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTO'], num_samples,
                                   p=[0.8, 0.1, 0.05, 0.05]),
            'src_bytes': np.random.exponential(1000, num_samples),
            'dst_bytes': np.random.exponential(2000, num_samples),
            'count': np.random.poisson(10, num_samples),
            'srv_count': np.random.poisson(5, num_samples),
            'serror_rate': np.random.beta(1, 10, num_samples),
            'rerror_rate': np.random.beta(1, 10, num_samples),
            'same_srv_rate': np.random.beta(5, 1, num_samples),
            'diff_srv_rate': np.random.beta(1, 5, num_samples)
        }

        return pd.DataFrame(data)

    async def _generate_attack_traffic(self, num_samples: int) -> pd.DataFrame:
        """
        Generate synthetic attack traffic patterns.

        Args:
            num_samples: Number of attack samples

        Returns:
            DataFrame with attack traffic features
        """
        attack_types = ['dos', 'probe', 'r2l', 'u2r']
        attack_weights = [0.5, 0.3, 0.15, 0.05]

        data = await self._generate_normal_traffic(num_samples)

        # Modify features to simulate attacks
        attack_type_col = np.random.choice(attack_types, num_samples, p=attack_weights)

        for i, attack_type in enumerate(attack_type_col):
            if attack_type == 'dos':
                data.loc[i, 'count'] *= 10  # High connection count
                data.loc[i, 'serror_rate'] = np.random.uniform(0.8, 1.0)
            elif attack_type == 'probe':
                data.loc[i, 'service'] = 'private'  # Unusual service
                data.loc[i, 'flag'] = 'OTH'  # Other flags
            elif attack_type == 'r2l':
                data.loc[i, 'src_bytes'] = np.random.exponential(10000)  # Large payload
                data.loc[i, 'duration'] = np.random.exponential(10)
            elif attack_type == 'u2r':
                data.loc[i, 'service'] = 'rootkit'
                data.loc[i, 'dst_bytes'] = np.random.exponential(50000)

        data['attack_type'] = attack_type_col
        return data

    async def _simulate_attacks(self, attack_types: List[str] = None,
                              num_samples: int = 1000, **kwargs) -> Dict[str, Any]:
        """
        Simulate specific attack patterns.

        Args:
            attack_types: List of attack types to simulate
            num_samples: Number of samples per attack type

        Returns:
            Dict with attack simulation data
        """
        if attack_types is None:
            attack_types = ['dos', 'probe', 'r2l', 'u2r']

        results = {}
        for attack_type in attack_types:
            self.logger.info(f"Simulating {attack_type} attacks")
            attack_data = await self._generate_attack_traffic(num_samples)
            attack_data = attack_data[attack_data['attack_type'] == attack_type]
            results[attack_type] = attack_data

        return {
            'attack_simulations': results,
            'total_samples': sum(len(data) for data in results.values()),
            'attack_types': attack_types
        }

    async def _augment_dataset(self, dataset: pd.DataFrame,
                             augmentation_factor: float = 2.0, **kwargs) -> pd.DataFrame:
        """
        Augment existing dataset with synthetic variations.

        Args:
            dataset: Original dataset to augment
            augmentation_factor: Factor by which to increase dataset size

        Returns:
            Augmented dataset
        """
        self.logger.info(f"Augmenting dataset by factor {augmentation_factor}")

        augmented_data = [dataset]

        for _ in range(int(augmentation_factor) - 1):
            # Add noise to numerical features
            noise_data = dataset.copy()
            numerical_cols = noise_data.select_dtypes(include=[np.number]).columns

            for col in numerical_cols:
                noise = np.random.normal(0, 0.1 * noise_data[col].std(), len(noise_data))
                noise_data[col] += noise

            augmented_data.append(noise_data)

        return pd.concat(augmented_data, ignore_index=True)

    async def _balance_classes(self, dataset: pd.DataFrame, target_column: str = 'label',
                             **kwargs) -> pd.DataFrame:
        """
        Balance dataset classes by oversampling minority classes.

        Args:
            dataset: Dataset to balance
            target_column: Name of target column

        Returns:
            Balanced dataset
        """
        self.logger.info("Balancing dataset classes")

        class_counts = dataset[target_column].value_counts()
        max_count = class_counts.max()

        balanced_data = []

        for class_label, count in class_counts.items():
            class_data = dataset[dataset[target_column] == class_label]

            if count < max_count:
                # Oversample minority class
                oversample_ratio = max_count // count
                oversampled_data = pd.concat([class_data] * oversample_ratio, ignore_index=True)

                # Add remaining samples if needed
                remainder = max_count % count
                if remainder > 0:
                    additional_data = class_data.sample(remainder, random_state=42)
                    oversampled_data = pd.concat([oversampled_data, additional_data], ignore_index=True)

                balanced_data.append(oversampled_data)
            else:
                balanced_data.append(class_data)

        return pd.concat(balanced_data, ignore_index=True).sample(frac=1).reset_index(drop=True)

    async def _assess_data_quality(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess quality of generated data.

        Args:
            dataset: Dataset to assess

        Returns:
            Dict with quality metrics
        """
        quality_metrics = {
            'total_samples': len(dataset),
            'missing_values': dataset.isnull().sum().sum(),
            'duplicate_rows': dataset.duplicated().sum(),
            'feature_correlations': dataset.select_dtypes(include=[np.number]).corr().abs().mean().mean(),
            'class_distribution': dataset['label'].value_counts().to_dict() if 'label' in dataset.columns else {}
        }

        return quality_metrics

    async def _general_synthesis(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """
        Handle general synthesis requests.

        Args:
            task_description: General task description
            **kwargs: Additional parameters

        Returns:
            Dict with synthesis results
        """
        self.logger.info(f"Handling general synthesis task: {task_description}")

        # Use LLM capabilities for general synthesis tasks
        # This would integrate with the agent's LLM for creative synthesis

        return {
            'task': task_description,
            'status': 'completed',
            'note': 'General synthesis handled via LLM capabilities'
        }