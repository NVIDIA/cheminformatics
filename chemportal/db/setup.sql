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
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(45) NOT NULL,
  `pipeline_user` VARCHAR(45) NOT NULL,
  `is_published` TINYINT NOT NULL,
  `definition` JSON NULL,
  `ui_resources` JSON NULL,
  `time_created` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `last_updated` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `is_deleted` TINYINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `pipeline_id_UNIQUE` (`id` ASC) VISIBLE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `cuchem_db`.`jobs`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `cuchem_db`.`jobs` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `pipeline_id` INT NOT NULL,
  `status` ENUM("successful", "in-progress", "errored") NOT NULL,
  `time_started` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `last_updated` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `job_configuration` TEXT NULL,
  PRIMARY KEY (`id`),
  INDEX `pipeline_id_idx` (`pipeline_id` ASC) VISIBLE,
  UNIQUE INDEX `id_UNIQUE` (`id` ASC) VISIBLE,
  CONSTRAINT `fk_pipeline_pipelineid`
    FOREIGN KEY (`pipeline_id`)
    REFERENCES `cuchem_db`.`pipelines` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `cuchem_db`.`job_tasks`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `cuchem_db`.`job_tasks` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `job_id` INT NOT NULL,
  `task_name` VARCHAR(45) NOT NULL,
  `status` ENUM("In-progess", "successful", "failed") NOT NULL,
  `time_started` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `last_updated` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `config` JSON NULL,
  `exit_code` INT NULL,
  `exit_message` TEXT NULL,
  PRIMARY KEY (`id`),
  INDEX `job_id_idx` (`job_id` ASC) VISIBLE,
  UNIQUE INDEX `id_UNIQUE` (`id` ASC) VISIBLE,
  CONSTRAINT `fk_job_tasks_jobid`
    FOREIGN KEY (`job_id`)
    REFERENCES `cuchem_db`.`jobs` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `cuchem_db`.`task_artifacts`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `cuchem_db`.`task_artifacts` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `task_id` INT NOT NULL,
  `name` VARCHAR(45) NOT NULL,
  `type` ENUM("computation") NOT NULL,
  `is_input` TINYINT NOT NULL,
  `artifact_value` TEXT NULL,
  PRIMARY KEY (`id`),
  INDEX `fk_job_artifacts_job_id_idx` (`task_id` ASC) VISIBLE,
  UNIQUE INDEX `artifact_id_UNIQUE` (`id` ASC) VISIBLE,
  CONSTRAINT `fk_task_artifacts_job_id`
    FOREIGN KEY (`task_id`)
    REFERENCES `cuchem_db`.`job_tasks` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
