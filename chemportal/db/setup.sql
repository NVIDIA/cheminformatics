-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema cuchem_db
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema cuchem_db
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `cuchem_db` DEFAULT CHARACTER SET utf8 ;
USE `cuchem_db` ;

-- -----------------------------------------------------
-- Table `cuchem_db`.`pipelines`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `cuchem_db`.`pipelines` (
  `pipeline_id` INT NOT NULL,
  `pipeline_name` VARCHAR(45) NOT NULL,
  `pipeline_user` VARCHAR(45) NOT NULL,
  `is_published` TINYINT NOT NULL,
  `pipeline_config` TEXT NULL,
  `pipeline_resources` TEXT NULL,
  PRIMARY KEY (`pipeline_id`),
  UNIQUE INDEX `pipeline_id_UNIQUE` (`pipeline_id` ASC) VISIBLE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `cuchem_db`.`jobs`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `cuchem_db`.`jobs` (
  `job_id` INT NOT NULL,
  `pipeline_id` INT NOT NULL,
  `job_status` ENUM("successful", "in-progress", "errored") NOT NULL,
  `time_started` DATETIME NOT NULL,
  `time_finished` DATETIME NULL,
  `job_configuration` TEXT NULL,
  PRIMARY KEY (`job_id`),
  INDEX `pipeline_id_idx` (`pipeline_id` ASC) VISIBLE,
  CONSTRAINT `fk_pipeline_pipelineid`
    FOREIGN KEY (`pipeline_id`)
    REFERENCES `cuchem_db`.`pipelines` (`pipeline_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `cuchem_db`.`job_tasks`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `cuchem_db`.`job_tasks` (
  `task_id` INT NOT NULL,
  `job_id` INT NOT NULL,
  `task_name` VARCHAR(45) NOT NULL,
  `task_status` ENUM("In-progess", "successful", "failed") NOT NULL,
  `time_started` DATETIME NOT NULL,
  `time_finished` DATETIME NULL,
  `task_config` TEXT NULL,
  `exit_code` INT NULL,
  `exit_message` TEXT NULL,
  PRIMARY KEY (`task_id`),
  INDEX `job_id_idx` (`job_id` ASC) VISIBLE,
  CONSTRAINT `fk_job_tasks_jobid`
    FOREIGN KEY (`job_id`)
    REFERENCES `cuchem_db`.`jobs` (`job_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `cuchem_db`.`job_artifacts`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `cuchem_db`.`job_artifacts` (
  `artifact_id` INT NOT NULL,
  `job_id` INT NOT NULL,
  `artifact_name` VARCHAR(45) NOT NULL,
  `artifact_type` ENUM("computation") NOT NULL,
  `is_input` TINYINT NOT NULL,
  `artifact_value` TEXT NULL,
  PRIMARY KEY (`artifact_id`),
  INDEX `job_id_idx` (`job_id` ASC) VISIBLE,
  CONSTRAINT `fk_job_artifacts_job_id`
    FOREIGN KEY (`job_id`)
    REFERENCES `cuchem_db`.`jobs` (`job_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
