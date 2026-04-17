// Standalone Terraform snippet to bootstrap Artifact Registry +
// the Cloud Build → Artifact Registry IAM binding. Apply once per
// project; the cloudbuild.yaml pipeline takes over from there.

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = { source = "hashicorp/google", version = ">= 5.30" }
  }
}

variable "project_id" { type = string }
variable "region"     { type = string, default = "us-central1" }

resource "google_artifact_registry_repository" "mldm" {
  project       = var.project_id
  location      = var.region
  repository_id = "mldm"
  description   = "Container images for the MLDM agent"
  format        = "DOCKER"

  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 30
    }
  }

  cleanup_policies {
    id     = "delete-untagged"
    action = "DELETE"
    condition {
      tag_state = "UNTAGGED"
      older_than = "604800s" // 7 days
    }
  }
}

data "google_project" "current" {
  project_id = var.project_id
}

// Cloud Build's default service account needs writer on the repo.
resource "google_artifact_registry_repository_iam_member" "cloudbuild_writer" {
  project    = var.project_id
  location   = google_artifact_registry_repository.mldm.location
  repository = google_artifact_registry_repository.mldm.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${data.google_project.current.number}@cloudbuild.gserviceaccount.com"
}
