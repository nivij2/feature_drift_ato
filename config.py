import hashlib
import json
import os
from collections import namedtuple
from pathlib import Path
from urllib.parse import urlparse

import boto3
from mergedeep import merge

from mlops.deployment.core import logging

AppConfig = namedtuple(
    'AppConfig',
    [
        'branch',
        'build_number',
        'commit_sha',
        'environment',
        'feature_engineering_jar',
        'fqn',
        'model',
        'pipeline',
        'project',
        'tenant',
        'debug'
    ]
)

TenantConfig = namedtuple(
    'TenantConfig',
    [
        'app_config',
        'aws_account_id',
        'aws_region',
        'build_artifacts',
        'build_number',
        'commit_sha',
        'crawler_role_arn',
        'domain_id',
        'emr_cluster_id',
        'environment',
        'environments',
        'execution_role_arn',
        'fqn',
        'lambda_role_arn',
        'lambda_security_group_ids',
        'lambda_subnet_ids',
        'mllib_image_uri',
        's3_bucket',
        'scheduler_role_arn',
        'studio_lifecycle_scripts',
        'studio_users',
        'tags',
        'tenant_name',
    ]
)

DeploymentConfig = namedtuple(
    'DeploymentConfig',
    TenantConfig._fields + (
        'branch',
        'branches',
        'feature_engineering_jar'
    )
)

ProjectConfig = namedtuple(
    'ProjectConfig',
    TenantConfig._fields + ('project_name',)
)

ModelConfig = namedtuple(
    'ModelConfig',
    ProjectConfig._fields + (
        'base_s3_path',
        'model_arn',
        'model_base_dir',
        'model_binaries_artifact',
        'model_binaries_dir',
        'model_binaries_location',
        'model_binary',
        'model_binary_sha',
        'model_code_artifact',
        'model_code_location',
        'model_code_sha',
        'model_description',
        'model_entrypoint',
        'model_group_arn',
        'model_image_uri',
        'model_image_file',
        'model_inference_config',
        'model_metadata',
        'model_name',
        'model_package_arn',
        'model_sha',
        'model_source_dir',
        'model_version',
    )
)

PipelineConfig = namedtuple(
    'PipelineConfig',
    ProjectConfig._fields + (
        'base_path',
        'base_s3_path',
        'feature_engineering_jar',
        'pipeline_module_location',
        'pipeline_module_name',
        'pipeline_name',
        'pipeline_name_prefix',
        'pipeline_schedule',
        'registered_model',
        'rule_name',
    )
)

GlueConfig = namedtuple(
    'PipelineConfig',
    ProjectConfig._fields + (
        'base_s3_path',
        'crawler_name_prefix',
        'datasets',
        'force_create_flag',
        'table_name_prefix',
    )
)

FeaturesConfig = namedtuple(
    'FeaturesConfig',
    ModelConfig._fields + (
        'features',
    )
)

ImageConfig = namedtuple(
    'ImageConfig',
    TenantConfig._fields + (
        'image_base_dir',
        'image_description',
        'image_name',
        'image_file',
        'image_uri'
    )
)


def config_attribute(nt, **kwargs):
    return nt._replace(**kwargs)


def config_dict(nt):
    return nt._asdict()


def read_config(path):
    return json.loads(Path(path).read_text())


def read_app_config(**kwargs):
    fqn_keys = ['tenant', 'project', 'model', 'pipeline', 'environment']

    return AppConfig(**merge(kwargs, {
        'debug': kwargs.get('debug', False),
        'model': kwargs.get('model'),
        'project': kwargs.get('project'),
        'pipeline': kwargs.get('pipeline'),
        'branch': kwargs.get('branch'),
        'feature_engineering_jar': kwargs.get('feature_engineering_jar'),
        'environment': kwargs['environment'].split('/')[-1],
        'fqn': "-".join([kwargs.get(k) for k in fqn_keys if kwargs.get(k)])
    }))


def read_tenant_config(app_config):
    tenant_spec = json.loads(Path(f"{app_config.tenant}/tenant.json").read_text())
    environment_spec = tenant_spec.get('environments', {}).get(app_config.environment, {})

    return TenantConfig(**merge(
        {
            'emr_cluster_id': None,
            'lambda_security_group_ids': None,
            'lambda_subnet_ids': None,
            'studio_lifecycle_scripts': None,
            'studio_users': None
        },
        tenant_spec,
        {
            'app_config': app_config,
            'fqn': app_config.fqn,
            'tenant_name': app_config.tenant,
            'environment': app_config.environment,
            'build_number': app_config.build_number,
            'commit_sha': app_config.commit_sha,
            'build_artifacts': './artifacts',
            'tags': {
                "Environment": app_config.environment,
                "BuildNumber": app_config.build_number,
                "CommitSha": app_config.commit_sha
            }
        },
        environment_spec
    ))


def read_deployment_config(app_config):
    tenant_config = read_tenant_config(app_config)
    deployment_spec = json.loads(Path(f"{app_config.tenant}/deployment.json").read_text())
    default_feature_engineering_jar = f"s3://{tenant_config.s3_bucket}/artifacts/feature-engineering-assembly-{app_config.commit_sha}.jar"

    return DeploymentConfig(**merge(
        config_dict(tenant_config),
        {
            'branch': app_config.branch,
            'feature_engineering_jar': app_config.feature_engineering_jar or default_feature_engineering_jar
        },
        deployment_spec
    ))


def read_project_config(app_config):
    tenant_config = read_tenant_config(app_config)
    project_spec = json.loads(Path(f"{app_config.tenant}/{app_config.project}/project.json").read_text())

    return ProjectConfig(**merge(
        config_dict(tenant_config),
        project_spec
    ))


def read_model_config(app_config):
    def model_code_hash(model_src_dir):
        sha1 = hashlib.sha1()

        if not os.path.exists(model_src_dir):
            sha1.update(''.encode("utf-8"))
            return sha1.hexdigest()

        model_files = list([fn for fn in os.listdir(model_src_dir)])
        model_files.sort()
        logging.info(f"Model files: {', '.join(model_files)}")

        for fn in model_files:
            p = os.path.join(model_src_dir, fn)
            if os.path.isfile(p):
                with open(p, 'rb') as f:
                    sha1.update(f.read())

        return sha1.hexdigest()

    def model_binary_hash(model_binaries_dir, model_file):
        sha1 = hashlib.sha1()
        with open(f"{model_binaries_dir}/{model_file}", 'rb') as f:
            sha1.update(f.read())
            return sha1.hexdigest()

    project_config = read_project_config(app_config)
    model_spec = json.loads(Path(f"{app_config.tenant}/{app_config.project}/{app_config.model}/model.json").read_text())

    model_code_sha = model_code_hash(
        f"{app_config.tenant}/{app_config.project}/{app_config.model}/{model_spec.get('model_source_dir', 'model')}"
    )

    if model_spec.get('model_binary'):
        model_binary_sha = model_binary_hash(
            f"{app_config.tenant}/{app_config.project}/{app_config.model}/{model_spec.get('model_binaries_dir', 'model')}",
            model_spec["model_binary"]
        )
    else:
        model_binary_sha = hashlib.sha1(''.encode("utf-8")).hexdigest()

    model_version_string = f"{model_binary_sha}{model_code_sha}{model_spec['model_version']}".encode("utf-8")
    model_sha = hashlib.sha1(model_version_string).hexdigest()
    base_s3_path = f"s3://{project_config.s3_bucket}/{app_config.tenant}-{app_config.project}-{app_config.model}-{app_config.environment}".replace("_", "-")

    return ModelConfig(**merge(
        config_dict(project_config),
        model_spec,
        {
            'model_arn': None,
            'model_code_artifact': None,
            'model_code_location': None,
            'model_binaries_artifact': None,
            'model_binaries_location': None,
            'model_package_arn': None,
            'model_group_arn': None,
            'model_sha': model_sha,
            'model_entrypoint': model_spec.get('model_entrypoint'),
            'model_image_uri': model_spec.get('model_image_uri'),
            'model_image_file': model_spec.get('model_image_file'),
            'model_binary': model_spec.get('model_binary'),
            'model_binary_sha': model_binary_sha,
            'model_code_sha': model_code_sha,
            'model_base_dir': f"{app_config.tenant}/{app_config.project}/{app_config.model}",
            'model_source_dir': f"{app_config.tenant}/{app_config.project}/{app_config.model}/{model_spec.get('model_source_dir', 'model')}",
            'model_binaries_dir': f"{app_config.tenant}/{app_config.project}/{app_config.model}/{model_spec.get('model_binaries_dir', 'model')}",
            'base_s3_path': base_s3_path
        }
    ))


def read_pipeline_config(app_config):
    base_path = f"{app_config.tenant}/{app_config.project}/{app_config.model}"
    module_name = f"{app_config.tenant}.{app_config.project}.{app_config.model}.{app_config.pipeline}"
    pipeline_name = f"{app_config.tenant}-{app_config.project}-{app_config.model}-{app_config.pipeline}-{app_config.environment}"
    base_s3_key = f"{app_config.tenant}-{app_config.project}-{app_config.model}-{app_config.environment}"

    project_config = read_project_config(app_config)
    pipeline_spec = json.loads(Path(f"{base_path}/pipeline.json").read_text())

    return PipelineConfig(**merge(
        config_dict(project_config),
        pipeline_spec,
        {
            'base_path': base_path,
            'base_s3_path': f"s3://{project_config.s3_bucket}/{base_s3_key}".replace("_", "-"),
            'feature_engineering_jar': app_config.feature_engineering_jar,
            'pipeline_name_prefix': pipeline_name.replace('_', '-'),
            'pipeline_name': pipeline_name.replace('_', '-'),
            'rule_name': pipeline_name.replace('_', '-'),
            'pipeline_module_location': f"./{base_path}/pipelines/{app_config.pipeline}.py",
            'pipeline_module_name': module_name,
            'pipeline_schedule': pipeline_spec['pipeline_schedule'].get(app_config.pipeline),
            'registered_model': None
        }
    ))


def read_dataset_config(app_config):
    base_path = f"{app_config.tenant}/{app_config.project}/{app_config.model}"

    project_config = read_project_config(app_config)
    glue_spec = json.loads(Path(f"{base_path}/dataset.json").read_text())
    model_spec = json.loads(Path(f"{base_path}/model.json").read_text())
    base_s3_key = f"{app_config.tenant}-{app_config.project}-{model_spec['model_name']}-{app_config.environment}"

    return GlueConfig(**merge(
        config_dict(project_config),
        glue_spec,
        {
            'base_s3_path': f"s3://{project_config.s3_bucket}/{base_s3_key}",
            'table_name_prefix': base_s3_key,
            'crawler_name_prefix': f"pi_risk_ml_{base_s3_key}",
            'force_create_flag': False
        }
    ))


def read_features_config(app_config):
    base_path = f"{app_config.tenant}/{app_config.project}/{app_config.model}"
    model_config = read_model_config(app_config)
    features_spec = json.loads(Path(f"{base_path}/features.json").read_text())

    return FeaturesConfig(**merge(
        config_dict(model_config),
        features_spec
    ))


def read_image_config(app_config):
    ##  We assume we only need one analytics docker, it might apply to each tenant
    docker_config_path = f"mllib/docker.json"
    tenant_config = read_tenant_config(app_config)

    docker_spec = json.loads(Path(docker_config_path).read_text())
    return ImageConfig(
        **merge(config_dict(tenant_config),
                docker_spec)
    )
